import torch

CONSTANTS = {
    'constant_2': lambda inputs: torch.tensor(2).float(),
    'constant_3': lambda inputs: torch.tensor(3).float(),
    'constant_4': lambda inputs: torch.tensor(4).float(),
    'constant_5': lambda inputs: torch.tensor(5).float(),
    'constant__1': lambda inputs: torch.tensor(-1).float(),
    'constant_1': lambda inputs: torch.tensor(1).float()
}