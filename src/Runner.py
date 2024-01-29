import torch
from abc import ABC, abstractmethod
from typing import Tuple, List, Literal
from e3nn.o3 import Irreps
import torch
from multiprocessing import cpu_count
import wandb
import torch.optim as optim
from torch_geometric.loader import DataLoader
from Dataset.CrystalSO3Dataset import CrystalSO3Dataset
from Model import SETransformer, SEMPNN, SEConv
import warnings
from tqdm import tqdm
from argparse import ArgumentParser
import json
from Model.BalancedIrreps import BalancedIrreps
from Dataset.Normalization import Normalizer

SEED = 13
NUM_WORKERS = min(cpu_count() - 1, 16)
PROJECT = "Transfer Learning For Crystal Properties"

class Runner(ABC):
    """
    Acts as a generic class for training and evaluating SEGNN models.
    """
    def __init__(self) -> None:
        """
        Initializes a runner with a given config. Takes arguments from commandline.

        Args:
            name: The name of the run for wandb
            config: The config to set up the model and run training with
            validation: Whether or not to do a validation run.
            graph_weights_from_pretrained: Graph weights to load the model with.
            finetunable: Whether or not to freeze the graph weights from pretrained.
        """
        super().__init__()

        args = ArgumentParser()
        args.add_argument("name", help="The name of the run", type=str)
        args.add_argument("config", help="The path to the config to run.", type=str)
        args.add_argument("--pretrained", help="The path to the pretrained weights", type=str)
        args.add_argument("--validation", help="If this is a validation run and not a testing run", action="store_true")
        args.add_argument("--finetunable", help="If the pretrained graph layers should be finetunable", action="store_true")
        args = args.parse_args()

        with open(args.config) as f:
            config = json.load(f)

        self.config = {
            **config,
            "seed": SEED,
            "pretrained": args.pretrained is not None,
            "validation": args.validation,
            "finetunable": args.finetunable
        }

        if args.pretrained is None and not args.finetunable:
            warnings.warn("Can't restrict finetuning if not providing a pretrained model, defaulting to training graph_nn.")
        
        wandb.login()
    
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(SEED)
        torch.backends.cudnn.benchmark = True

        print("loading data")
        train_dataset, validation_dataset = self.setup_datasets()
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
        self.setup_model(self.config, num_atom_feats)
        if args.pretrained:
            with open(args.pretrained) as f:
                self.model.graph_nn.load_state_dict(json.load(f))
            if not args.finetunable:
                for param in self.model.graph_nn.parameters:
                    param.requires_grad = False
        print("loaded model")

        self.model = self.model.to(self.device)
        self.setup_optimizer()
        self.setup_scheduler()

        # Setup Normalizer for unnorming later
        with open(f"{self.default_data_path}/coef_stats.json") as f:
            self.normalizer = Normalizer(json.load(f), self.device)

        self.name = args.name
        self.best_logs = None
        self.best_epoch = None
        self.best_model_state_dict = None

    def main(self):
        """
        The main file which runs the training for a model.
        """
        tags = [self.config["model"], "validation" if self.config["validation"] else "testing", 
                "normalized" if self.config["normalize"] else "unnormalized"]
        with wandb.init(config=self.config, name=self.name, project=PROJECT, tags=tags):
            for epoch in tqdm(range(self.config["epochs"]), "Epoch"):
                logs = {}
                logs.update({"learning_rate": self.scheduler.get_last_lr()[0]})
                logs.update(self.run_epoch(self.train_loader, train=True))
                logs.update(self.run_epoch(self.validation_loader, train=False))
                wandb.log(logs)
                self.scheduler.step()
                self.record_best(epoch, logs)
        self.log_best()

    def setup_datasets(self) -> Tuple[CrystalSO3Dataset, CrystalSO3Dataset]:
        """
        Sets up the dataset for training

        Returns:
            The training dataset and the testing dataset
        """
        with open(f"{self.default_data_path}/train/data.json") as train_data:
            train_data = json.load(train_data)
        with open(f"{self.default_data_path}/validate/data.json") as validation_data:
            if self.config["validation"]:
                test_data = json.load(validation_data)
            else:
                train_data.extend(json.load(validation_data))
                with open(f"{self.default_data_path}/test/data.json") as test_data:
                    test_data = json.load(test_data)
        with open(f"./Dataset/atom_init.json") as atom_init:
            atom_init = json.load(atom_init)
        return (self.dataset_class(train_data, atom_init, self.config["lmax_h"], radius=self.config["radius"], 
                                   max_neighbors=self.config["max_neighbors"], normalize=self.config["normalize"]), 
                self.dataset_class(test_data, atom_init, self.config["lmax_h"], radius=self.config["radius"], 
                                   max_neighbors=self.config["max_neighbors"], normalize=self.config["normalize"]))

    @property
    @abstractmethod
    def default_data_path(self) -> str:
        """
        The location where all the data for this runner is stored. i.e Dataset/elastic
        """
        raise NotImplementedError("Set default data path in subclasses.")
    
    @abstractmethod
    def dataset_class(self, data: List[dict], atom_init: dict, lmax_h: int, radius: float, max_neighbors: int, normalize: bool) -> CrystalSO3Dataset:
        """
        Abstract factory for the dataset class. Takes in same arguments as a CrystalSO3Dataset and applies
        to a specific class
        """
        raise NotImplementedError("Set up which dataset class you want to use in a subclass")

    @abstractmethod
    def run_epoch(self, dataloader: DataLoader, train: bool):
        """
        Runs a single epoch 

        Args:
            dataloader: The dataloader to use
            train: Whether or not to run this as a training loop.
        """
        raise NotImplementedError("run_epoch must be implemented in subclasses.")
    
    @abstractmethod
    def record_best(self, epoch: int, logs: dict):
        """
        Records best result and model to be logged later

        Args:
            epoch: The epoch number that these logs are associated with
            logs: The logs to check if they are the best
        """
        raise NotImplementedError("record_best must be set up in subclasses.")
    
    def log_best(self):
        """
        Logs best results to wandb
        """
        for k, v in self.best_logs.items():
            wandb.summary[f"Best {k}"] = v
        wandb.summary["Best Epoch"] = self.best_epoch
    
    @property
    @abstractmethod
    def output_irreps_lst(self) -> List[Irreps]:
        """
        Defines the model output
        """
        raise NotImplementedError("Must choose output in subclasses")
    
    def setup_model(self, config: dict, num_atom_feats: int):
        """
        Sets up the models to be run with the right output style

        Args:
            config: The config file to use when setting up the models
            num_atom_feats: The number of atom features to use.

        Returns:
            The requested model.
        """
        if "segnn" == config["model"]:
            self.model = self.setup_sempgnn(num_atom_feats=num_atom_feats, lmax_h=config["lmax_h"], 
                                hidden_features=config["hidden_features"], num_graph_layers=config["num_graph_layers"],
                                num_prepool_layers=config["num_prepool_layers"], num_postpool_layers=config["num_postpool_layers"],
                                output_irreps_lst=self.output_irreps_lst, norm=config["layer_norm"], pooling=config["pooling"])
        elif "seconv" in config["model"]:
            self.model = self.setup_seconv(num_atom_feats=num_atom_feats, lmax_h=config["lmax_h"], hidden_features=config["hidden_features"], 
                                num_graph_layers=config["num_graph_layers"], num_prepool_layers=config["num_prepool_layers"], 
                                num_postpool_layers=config["num_postpool_layers"], conv_type=config["model"][config["model"].rfind("-") + 1:],
                                output_irreps_lst=self.output_irreps_lst, norm=config["layer_norm"], pooling=config["pooling"])
        elif "setransformer" == config["model"]:
            self.model = self.setup_setransformer(num_atom_feats=num_atom_feats, lmax_h=config["lmax_h"], 
                                hidden_features=config["hidden_features"], num_graph_layers=config["num_graph_layers"],
                                num_prepool_layers=config["num_prepool_layers"], num_postpool_layers=config["num_postpool_layers"],
                                output_irreps_lst=self.output_irreps_lst, norm=config["layer_norm"], max_radius=config["radius"], 
                                pooling=config["pooling"])
        else:
            raise Exception("No such model of that type")
            
    def setup_irreps(self, num_atom_feats: int, lmax_h: int, hidden_features: int) -> Tuple[Irreps, Irreps, Irreps, Irreps, Irreps, Irreps, Irreps]:
        """
        Sets up the irreps for the model input, hidden layers, and output

        Args:
            num_atom_feats: The number of atom scalar features provided.
            lmax_h: The maximum l for the hidden layers
            hidden_features: The number of hidden features

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
        hidden_irreps = BalancedIrreps(lmax_h, hidden_features)
        return input_irreps, edge_attr_irreps, node_attr_irreps, additional_message_irreps, hidden_irreps

    def setup_sempgnn(self, num_atom_feats: int, lmax_h: int, hidden_features: int, num_graph_layers: int, 
                    num_prepool_layers: int, num_postpool_layers: int, output_irreps_lst: List[Irreps],
                    norm: Literal["none", "batch"], pooling: Literal["mean", "add", "max"]) -> SEMPNN:
        """
        Sets up a SEMPN model based on the parameters provided.

        Args:
            num_atom_feats: How many atomic features will be provided
            lmax_h: The maximum l for the hidden layer
            hidden_features: How many features in the hidden layers
            num_graph_layers: How many hidden layers of graph interaction
            num_prepool_layers: How many linear layers on just node values
            num_postpool_layers: How many layers before final output after pooling
            output_irreps_lst: The list of outputs of the model
            norm: The layer normalization the model may employ
            pooling: How to pool the nodes together

        Returns the specified SEMPNN model
        """
        (input_irreps, 
        edge_attr_irreps, 
        node_attr_irreps, 
        additional_message_irreps, 
        hidden_irreps) = self.setup_irreps(num_atom_feats, lmax_h, hidden_features)

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

    def setup_seconv(self, num_atom_feats: int, lmax_h: int, hidden_features: int, num_graph_layers: int, 
                    num_prepool_layers: int, num_postpool_layers: int, 
                    conv_type: Literal["linear", "nonlinear"], output_irreps_lst: List[Irreps],
                    norm: Literal["none", "batch"], pooling: Literal["mean", "add", "max"]) -> SEConv:
        """
        Sets up a SEConv model based on the parameters provided.

        Args:
            num_atom_feats: How many atomic features will be provided
            lmax_h: The maximum l for the hidden layer
            hidden_features: How many features in the hidden layers
            num_graph_layers: How many hidden layers of graph interaction
            num_prepool_layers: How many linear layers on just node values
            num_postpool_layers: How many layers before final output after pooling
            conv_type: Either linear or nonlinear convolutions
            output_irreps_lst: The list of outputs of the model
            norm: The layer normalization the model may employ
            pooling: How to pool the nodes together

        Returns the specified SEConv model.
        """
        (input_irreps, 
        edge_attr_irreps, 
        node_attr_irreps, 
        additional_message_irreps, 
        hidden_irreps) = self.setup_irreps(num_atom_feats, lmax_h, hidden_features)

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

    def setup_setransformer(self, num_atom_feats: int, lmax_h: int, hidden_features: int, num_graph_layers: int, 
                    num_prepool_layers: int, num_postpool_layers: int, output_irreps_lst: List[Irreps],
                    norm: Literal["none", "batch"], max_radius: float, pooling: Literal["mean", "add", "max"]) -> SETransformer:
        """
        Sets up a SETransformer model based on the parameters provided.

        Args:
            num_atom_feats: How many atomic features will be provided
            lmax_h: The maximum l for the hidden layer
            hidden_features: How many features in the hidden layers
            num_graph_layers: How many hidden layers of graph interaction
            num_prepool_layers: How many linear layers on just node values
            num_postpool_layers: How many layers before final output after pooling
            output_irreps_lst: The list of outputs of the model
            norm: The layer normalization the model may employ
            max_radius: The maximum radius for cutoff function
            pooling: How to pool the nodes together

        Returns the SETransformer model as specified
        """
        (input_irreps, 
        edge_attr_irreps, 
        node_attr_irreps, 
        additional_message_irreps, 
        hidden_irreps) = self.setup_irreps(num_atom_feats, lmax_h, hidden_features)

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
    
    def setup_optimizer(self):
        """
        Sets up an optimizer for a model based on the config
        """
        if self.config["optimizer"] == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.config["learning_rate"], 
                                       momentum=self.config["momentum"], weight_decay=self.config["weight_decay"])
        elif self.config["optimizer"] == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["learning_rate"], 
                                        weight_decay=self.config["weight_decay"])
        else:
            raise Exception("Not a legal optimizer")

    def setup_scheduler(self):
        """
        Sets up an optimizer scheduler based on the config
        """
        if "setransformer" == self.config["model"]:
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=self.config["epochs"], T_mult=1, eta_min=1e-8)
        else:
            # Steereable Equivariant
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, gamma=0.1, milestones=[0.8*self.config["epochs"], 0.9*self.config["epochs"]])