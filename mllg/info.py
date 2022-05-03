import time
from typing import Any
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class TestInfo:
    loss_type: str
    loss: float
    time: float = 0

    def __post_init__(self):
        
        # TODO: This is very bad
        if self.time == 0:
            super().__setattr__("time", time.time())
    
    @classmethod
    def from_dict(cls, data):
        return cls(loss_type=data['loss_type'], loss=data['loss'], time=data['time'])

    def __str__(self):
        dict_val = asdict(self)
        dict_val["type"] = "test_info"
        return str(dict_val)

    def dump(self, file_obj):
        file_obj.write(str(self) + "\n")


@dataclass(frozen=True)
class LossInfo:
    loss_type: str
    loss: float
    time: float = 0

    def __post_init__(self):
        
        # TODO: This is very bad
        if self.time == 0:
            super().__setattr__("time", time.time())
    
    @classmethod
    def from_dict(cls, data):
        return cls(loss_type=data['loss_type'], loss=data['loss'], time=data['time'])

    def __str__(self):
        dict_val = asdict(self)
        dict_val["type"] = "loss_info"
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
        if self.time == 0:
            super().__setattr__("time", time.time())

    @classmethod
    def from_dict(cls, data):
        losses = [TestInfo.from_dict(l) for l in data['losses']]
        return cls(epoch=data['epoch'], batch_idx=data['batch_idx'],  losses=losses, time=data['time'])

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
        if self.time == 0:
            super().__setattr__("time", time.time())

    @classmethod
    def from_dict(cls, data):
        return cls(epoch=data['epoch'], batch_idx=data['batch_idx'], loss=data['loss'], time=data['time'])

    def __str__(self):
        dict_val = asdict(self)
        dict_val["type"] = "train_info"
        return str(dict_val)

    def dump(self, file_obj):
        file_obj.write(str(self) + "\n")

@dataclass(frozen=True)
class TrainStepInfo:
    epoch: int
    batch_idx: int
    losses: Any
    time: float = 0

    def __post_init__(self):
        if self.time == 0:
            super().__setattr__("time", time.time())

    @classmethod
    def from_dict(cls, data):
        losses = [LossInfo.from_dict(l) for l in data['losses']]
        return cls(epoch=data['epoch'], batch_idx=data['batch_idx'],  losses=losses, time=data['time'])

    def __str__(self):
        dict_val = asdict(self)
        dict_val["type"] = "trainstep_info"
        return str(dict_val)

    def dump(self, file_obj):
        file_obj.write(str(self) + "\n")
