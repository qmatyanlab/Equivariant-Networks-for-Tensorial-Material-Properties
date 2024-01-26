import torch
from abc import ABC, abstractmethod
from typing import Optional

class Trainer(ABC):
    """
    Acts as a generic class for training and evaluating SEGNN models.
    """

    def __init__(self, validation: bool, from_pretrained: Optional[dict], finetunable: bool) -> None:
        super().__init__()

    @abstractmethod
    def run_epoch():
        raise NotImplementedError("run_epoch must be implemented in subclasses.")