
import ast
from dataclasses import dataclass
from typing import Dict, List
from mllg.info import TrainInfo, ValidationInfo

@dataclass(frozen=True)
class RunSummary:
    config: Dict
    train_steps: List[TrainInfo]
    val_steps: List[ValidationInfo]

    @classmethod
    def from_file(cls, file_path):
        with open(file_path, 'r') as file_data:
            lines = file_data.readlines()
        
        train_steps = []
        val_steps = []
        for l in lines:
            try:
                data = ast.literal_eval(l)
            except SyntaxError:
                pass

            if data['type'] == 'config':
                config = data
            elif data['type'] == 'train_info':
                train_steps.append(TrainInfo.from_dict(data))
            elif data['type'] == 'val_info':
                val_steps.append(ValidationInfo.from_dict(data))

        return RunSummary(config, train_steps, val_steps)
    
    def train_loss_vals(self):
        loss_vals = [t.loss for t in self.train_steps]
        return loss_vals

    def train_loss_batches(self, batch_size):
        batch_nums = [batch_size * (i + 1) for i in range(len(self.train_steps))]
        return batch_nums

    def validation_vals(self, val_name):
        vals = []
        for val in self.val_steps:
            for loss in val.losses:
                print(loss)
                if loss.loss_type == val_name:
                    vals.append(loss.loss)
        return vals

