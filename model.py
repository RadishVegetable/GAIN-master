import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, dim):
        super(Generator, self).__init__()
        self.sequence = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(), nn.Linear(dim, dim), nn.ReLU(),
                                      nn.Linear(dim, dim), nn.Sigmoid())

    def forward(self, new_x, m):
        inputs = torch.cat(dim=1, tensors=[new_x, m])
        out = self.sequence(inputs)
        return out


class Discriminator(nn.Module):
    def __init__(self, dim):
        super(Discriminator, self).__init__()
        self.sequence = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(), nn.Linear(dim, dim), nn.ReLU(),
                                      nn.Linear(dim, dim), nn.Sigmoid())
    def forward(self, new_x, h):
        inputs = torch.cat(dim=1, tensors=[new_x, h])
        return self.sequence(inputs)