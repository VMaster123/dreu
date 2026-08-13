import torch
import torch.nn as nn


class SurrogateModel(nn.Module):

    def __init__(
        self,
        input_dim=64,
    ):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                input_dim,
                128,
            ),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(
                128,
                64,
            ),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(
                64,
                1,
            ),
        )

    def forward(self, x):

        return self.network(x)
