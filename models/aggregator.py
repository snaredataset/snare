import torch
import torch.nn as nn


class MaxPool(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, x):
        x, _ = x.max(dim=-2) # [B 14 512] -> [B 512]
        return x


class MeanPool(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, x):
        return x.mean(dim=-2) # [B 14 512] -> [B 512]


class RandomIndex(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, x):
        batch_idxs = torch.randint(x.shape[1], (x.shape[0],))               # [B]
        return x[torch.arange(0, x.shape[0], dtype=torch.long), batch_idxs] # [B 512]


class TwoRandomIndex(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, x):
        batch_idxs_1 = torch.randint(x.shape[1], (x.shape[0],))              # [B]
        x1 = x[torch.arange(0, x.shape[0], dtype=torch.long), batch_idxs_1]  # [B 512]
        batch_idxs_2 = torch.randint(x.shape[1], (x.shape[0],))              # [B]
        x2 = x[torch.arange(0, x.shape[0], dtype=torch.long), batch_idxs_2]  # [B 512]
        x, _ = torch.stack([x1, x2], dim=-1).max(dim=-1)                     # [B 512]
        return x


names = {
    'meanpool': MeanPool,
    'maxpool': MaxPool,
    'random_index': RandomIndex,
    'two_random_index': TwoRandomIndex,
}