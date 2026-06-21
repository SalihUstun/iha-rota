import torch
import torch.nn as nn
from config import OBS_DIM, ACTION_DIM


class IhaPolicy(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(OBS_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, ACTION_DIM),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.fc(x)
