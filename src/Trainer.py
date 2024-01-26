import torch
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Literal
from e3nn.o3 import Irreps
import torch
from multiprocessing import cpu_count
import wandb
import torch.optim as optim
from torch_geometric.loader import DataLoader
from Dataset.CrystalSO3Dataset import CrystalSO3Dataset
from Model import SEGNN, SETransformer, SEMPNN, SEConv
import warnings
from math import inf
from tqdm import tqdm

SEED = 13
NUM_WORKERS = min(cpu_count() - 1, 16)
PROJECT = "Transfer Learning For Crystal Properties"

class Trainer(ABC):
    """
    Acts as a generic class for training and evaluating SEGNN models.
    """
    def __init__(self, config: dict, validation: bool, graph_weights_from_pretrained: Optional[dict], finetunable: bool) -> None:
        """
        Initializes a Trainer with a given config

        Args:
            config: The config to set up the model and run training with
            validation: Whether or not to do a validation run.
            graph_weights_from_pretrained: Graph weights to load the model with
        """
        super().__init__()

        if graph_weights_from_pretrained is None and not finetunable:
            warnings.warn("Can't restrict finetuning if not providing a pretrained model.")
        
        wandb.login()
    
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(SEED)
        torch.backends.cudnn.benchmark = True

        print("loading data")
        train_dataset, validation_dataset = self.setup_datasets(validation=validation, radius=config["radius"], lmax_h=config["lmax_h"],
                                                        normalize=config["normalize"], max_neighbors=config["max_neighbors"])
        num_atom_feats = train_dataset[0].x.shape[-1] 

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=NUM_WORKERS,
            drop_last=False,
            pin_memory=False,
            generator=torch.Generator().manual_seed(SEED)
        )

        self.validation_loader = DataLoader(
            validation_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=NUM_WORKERS,
            drop_last=False,
            pin_memory=False
        )
        print("loaded data")

        print("loading model")
        self.model = self.setup_model(config, num_atom_feats)
        if graph_weights_from_pretrained:
            self.model.graph_nn.load_state_dict(graph_weights_from_pretrained)
            if not finetunable:
                for param in self.model.graph_nn.parameters:
                    param.requires_grad = False
        print("loaded model")

        self.model = self.model.to(self.device)

        # Setup optimizer and scheduled learning rate
        self.optimizer = self.setup_optimizer(config, self.model)
        self.scheduler = self.setup_scheduler(config, self.optimizer)

        self.config = {
            **config,
            "seed": SEED,
            "validation": validation
        }

    def main(self, name: str):
        tags = [self.config["model"], "validation" if self.config["validation"] else "testing", "normalized" if self.config["normalize"] else "unnormalized"]
        with wandb.init(config=self.config, name=name, project=PROJECT, tags=tags):
            for epoch in tqdm(range(self.config["epochs"]), "Epoch"):
                logs = {}
                logs.update({"learning_rate": self.scheduler.get_last_lr()[0]})
                logs.update(self.run_epoch(train=True))
                logs.update(self.run_epoch(train=False))
                wandb.log(logs)
                self.scheduler.step()
                self.record_best(epoch, logs)

    @abstractmethod
    def run_epoch(self):
        raise NotImplementedError("run_epoch must be implemented in subclasses.")
    
    @abstractmethod
    def setup_datasets(self, validation: bool, radius: float = 8.0, normalize: bool = True, 
                       lmax_h: int = 4, max_neighbors: int = 12) -> Tuple[CrystalSO3Dataset, CrystalSO3Dataset]:
        raise NotImplementedError("setup_dataests must be implemented in subclasses.")
    
    @abstractmethod
    def setup_output(self):
        raise NotImplementedError("setup_output must be implemented in subclasses")
    
    @abstractmethod
    def record_best(self, epoch, logs):
        raise NotImplementedError("record_best must be set up in subclasses.")
    
    @abstractmethod
    @property
    def output_irreps_lst(self) -> List[Irreps]:
        raise NotImplementedError("Must choose output ")
    
    def setup_model(self, config: dict, num_atom_feats: int) -> SEGNN:
        """
        Sets up the models to be run with the right output style

        Args:
            config: The config file to use when setting up the models
            num_atom_feats: The number of atom features to use.

        Returns:
            The requested model.
        """
        # Setup specific model
        if "segnn" == config["model"]:
            model = self.setup_sempnn(num_atom_feats=num_atom_feats, lmax_h=config["lmax_h"], 
                                hidden_irreps=config["hidden_irreps"], num_graph_layers=config["num_graph_layers"],
                                num_prepool_layers=config["num_prepool_layers"], num_postpool_layers=config["num_postpool_layers"],
                                output_irreps_lst=self.output_irreps_lst, norm=config["layer_norm"], pooling=config["pooling"])
        elif "seconv" in config["model"]:
            model = self.setup_seconv(num_atom_feats=num_atom_feats, lmax_h=config["lmax_h"], hidden_irreps=config["hidden_irreps"], 
                                num_graph_layers=config["num_graph_layers"], num_prepool_layers=config["num_prepool_layers"], 
                                num_postpool_layers=config["num_postpool_layers"], conv_type=config["model"][config["model"].rfind("-") + 1:],
                                output_irreps_lst=self.output_irreps_lst, norm=config["layer_norm"], pooling=config["pooling"])
        elif "setransformer" == config["model"]:
            model = self.setup_setransformer(num_atom_feats=num_atom_feats, lmax_h=config["lmax_h"], 
                                hidden_irreps=config["hidden_irreps"], num_graph_layers=config["num_graph_layers"],
                                num_prepool_layers=config["num_prepool_layers"], num_postpool_layers=config["num_postpool_layers"],
                                output_irreps_lst=self.output_irreps_lst, norm=config["layer_norm"], max_radius=config["radius"], 
                                pooling=config["pooling"])
        else:
            raise Exception("No such model of that type")
        return model
            
    def setup_irreps(self, num_atom_feats: int, lmax_h: int, hidden_irreps: str) -> Tuple[Irreps, Irreps, Irreps, Irreps, Irreps, Irreps, Irreps]:
        """
        Sets up the irreps for the model input, hidden layers, and output

        Args:
            num_atom_feats: The number of atom scalar features provided.
            lmax_h: The maximum l for the hidden layers
            hidden_irreps: The hidden irreps string

        Returns all the required irreps
        """
        # Scalar one hot encoding
        input_irreps = Irreps([(num_atom_feats, (0, 1))])
        # Irreps from relative position between atoms based on spherical harmonics
        edge_attr_irreps = Irreps.spherical_harmonics(lmax_h)
        node_attr_irreps = Irreps.spherical_harmonics(lmax_h)
        # Distance is a singular scalar
        additional_message_irreps = Irreps("1x0e")
        # Create hidden irreps
        hidden_irreps = Irreps(hidden_irreps)
        return input_irreps, edge_attr_irreps, node_attr_irreps, additional_message_irreps, hidden_irreps

    def setup_sempnn(self, num_atom_feats: int, lmax_h: int, hidden_irreps: str, num_graph_layers: int, 
                    num_prepool_layers: int, num_postpool_layers: int, output_irreps_lst: List[Irreps],
                    norm: Literal["none", "batch"], pooling: Literal["mean", "add", "max"]) -> SEMPNN:
        """
        Sets up a SEMPNN model based on the parameters provided.

        Args:
            num_atom_feats: How many atomic features will be provided
            lmax_h: The maximum l for the hidden layer
            hidden_irreps: The hidden irreps string
            num_graph_layers: How many hidden layers of graph interaction
            num_prepool_layers: How many linear layers on just node values
            num_postpool_layers: How many layers before final output after pooling
            output_irreps_lst: The list of outputs of the model
            norm: The layer normalization the model may employ
            pooling: How to pool the nodes together
            subspace_type: How to allocate the irreps used in the hidden layers.

        Returns the specified SEMPNN model
        """
        from Model.SEMPNN import SEMPNN
        (input_irreps, 
        edge_attr_irreps, 
        node_attr_irreps, 
        additional_message_irreps, 
        hidden_irreps) = self.setup_irreps(num_atom_feats, lmax_h, hidden_irreps)

        return SEMPNN(input_irreps,
            hidden_irreps,
            output_irreps_lst,
            edge_attr_irreps,
            node_attr_irreps,
            num_graph_layers=num_graph_layers,
            num_prepool_layers=num_prepool_layers,
            num_postpool_layers=num_postpool_layers,
            additional_message_irreps=additional_message_irreps,
            norm=norm,
            pooling=pooling)

    def setup_seconv(self, num_atom_feats: int, lmax_h: int, hidden_irreps: str, num_graph_layers: int, 
                    num_prepool_layers: int, num_postpool_layers: int, 
                    conv_type: Literal["linear", "nonlinear"], output_irreps_lst: List[Irreps],
                    norm: Literal["none", "batch"], pooling: Literal["mean", "add", "max"]) -> SEConv:
        """
        Sets up a SEConv model based on the parameters provided.

        Args:
            num_atom_feats: How many atomic features will be provided
            lmax_h: The maximum l for the hidden layer
            hidden_irreps: How many features in the hidden layers
            num_graph_layers: How many hidden layers of graph interaction
            num_prepool_layers: How many linear layers on just node values
            num_postpool_layers: How many layers before final output after pooling
            conv_type: Either linear or nonlinear convolutions
            output_irreps_lst: The list of outputs of the model
            norm: The layer normalization the model may employ
            pooling: How to pool the nodes together
            subspace_type: How to allocate the irreps used in the hidden layers.

        Returns the specified SEConv model.
        """
        from Model.SEConvNN import SEConv
        (input_irreps, 
        edge_attr_irreps, 
        node_attr_irreps, 
        additional_message_irreps, 
        hidden_irreps) = self.setup_irreps(num_atom_feats, lmax_h, hidden_irreps)

        return SEConv(input_irreps,
            hidden_irreps,
            output_irreps_lst,
            edge_attr_irreps,
            node_attr_irreps,
            num_graph_layers=num_graph_layers,
            num_prepool_layers=num_prepool_layers,
            num_postpool_layers=num_postpool_layers,
            additional_message_irreps=additional_message_irreps,
            conv_type=conv_type,
            norm=norm,
            pooling=pooling)

    def setup_setransformer(self, num_atom_feats: int, lmax_h: int, hidden_irreps: str, num_graph_layers: int, 
                    num_prepool_layers: int, num_postpool_layers: int, output_irreps_lst: List[Irreps],
                    norm: Literal["none", "batch"], max_radius: float, pooling: Literal["mean", "add", "max"]) -> SETransformer:
        """
        Sets up a SETransformer model based on the parameters provided.

        Args:
            num_atom_feats: How many atomic features will be provided
            lmax_h: The maximum l for the hidden layer
            hidden_irreps: How many features in the hidden layers
            num_graph_layers: How many hidden layers of graph interaction
            num_prepool_layers: How many linear layers on just node values
            num_postpool_layers: How many layers before final output after pooling
            output_irreps_lst: The list of outputs of the model
            norm: The layer normalization the model may employ
            max_radius: The maximum radius for cutoff function
            pooling: How to pool the nodes together
            subspace_type: How to allocate the irreps used in the hidden layers.

        Returns the SETransformer model as specified
        """
        from Model.SETransformer import SETransformer
        (input_irreps, 
        edge_attr_irreps, 
        node_attr_irreps, 
        additional_message_irreps, 
        hidden_irreps) = self.setup_irreps(num_atom_feats, lmax_h, hidden_irreps)

        return SETransformer(input_irreps,
            hidden_irreps,
            output_irreps_lst,
            edge_attr_irreps,
            node_attr_irreps,
            num_graph_layers=num_graph_layers,
            num_prepool_layers=num_prepool_layers,
            num_postpool_layers=num_postpool_layers,
            additional_message_irreps=additional_message_irreps,
            norm=norm,
            max_radius=max_radius,
            pooling=pooling)
    
    def setup_optimizer(self, config: dict, model: torch.nn.Module):
        """
        Sets up an optimizer for a model based on the config

        Args:
            config: The config to pull optimizer info from
            model: The model to build the optimizer for

        Returns:
            The optimizer as specified by the config for the model.
        """
        if config["optimizer"] == "sgd":
            self.optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=config["momentum"], weight_decay=config["weight_decay"])
        elif config["optimizer"] == "adam":
            self.optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        else:
            raise Exception("Not a legal optimizer")

    def setup_scheduler(self, config: dict, optimizer: optim.Optimizer):
        """
        Sets up an optimizer scheduler based on the config

        Args:
            config: The config to pull scheduler information from
            optimizer: The optimizer to build the scheduler around
        """
        if "setransformer" == config["model"]:
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config["epochs"], T_mult=1, eta_min=1e-8)
        else:
            # Steereable Equivariant
            self.scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.1, milestones=[0.8*config["epochs"], 0.9*config["epochs"]])