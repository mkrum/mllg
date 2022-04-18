import time
from typing import Any
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class TestInfo:
    loss_type: str
    loss: float
    time: float = 0

    def __post_init__(self):
        super().__setattr__("time", time.time())

    def __str__(self):
        dict_val = asdict(self)
        dict_val["type"] = "test_info"
        return str(dict_val)

    def dump(self, file_obj):
        file_obj.write(str(self) + "\n")


@dataclass(frozen=True)
class ValidationInfo:
    epoch: int
    batch_idx: int
    losses: Any
    time: float = 0

    def __post_init__(self):
        super().__setattr__("time", time.time())

    def __str__(self):
        dict_val = asdict(self)
        dict_val["type"] = "val_info"
        return str(dict_val)

    def dump(self, file_obj):
        file_obj.write(str(self) + "\n")


@dataclass(frozen=True)
class TrainInfo:
    epoch: int
    batch_idx: int
    loss: float
    time: float = 0

    def __post_init__(self):
        super().__setattr__("time", time.time())

    def __str__(self):
        dict_val = asdict(self)
        dict_val["type"] = "train_info"
        return str(dict_val)

    def dump(self, file_obj):
        file_obj.write(str(self) + "\n")
