import sys
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(".."))

from dataset_loader import QuantumDataset

TRAIN_DATA = "../data/random_train.pkl"
VAL_DATA = "../data/random_val.pkl"

BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3


class BaselineMLP(nn.Module):
    """
    Direct regression baseline.

    Input:
        [classical_features | theta]

    Output:
        normalized energy
    """

    def __init__(self, input_dim=35):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def train():

    train_dataset = QuantumDataset(TRAIN_DATA)

    # get normalization stats
    energy_mean = train_dataset.energy_mean
    energy_std = train_dataset.energy_std

    val_dataset = QuantumDataset(
        VAL_DATA, energy_mean=energy_mean, energy_std=energy_std
    )

    # ==========================
    # OVERFIT DEBUG TEST
    # ==========================

    train_dataset = torch.utils.data.Subset(train_dataset, range(128))

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    #
    # Infer input dimension automatically
    #

    classical, theta, energy = train_dataset[0]

    input_dim = classical.numel() + theta.numel()

    print(f"Input dimension = {input_dim}")

    model = BaselineMLP(input_dim=input_dim)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):

        ###################
        # TRAIN
        ###################

        model.train()

        train_loss = 0.0

        for classical, theta, energy in train_loader:

            #
            # IMPORTANT
            #
            # Ensure target shape matches prediction
            #

            if energy.ndim == 1:
                energy = energy.unsqueeze(1)

            x = torch.cat([classical, theta], dim=1)

            prediction = model(x)

            loss = criterion(prediction, energy)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        ###################
        # VALIDATION
        ###################

        model.eval()

        val_loss = 0.0

        pred_std = []
        target_std = []

        with torch.no_grad():

            for classical, theta, energy in val_loader:

                if energy.ndim == 1:
                    energy = energy.unsqueeze(1)

                x = torch.cat([classical, theta], dim=1)

                prediction = model(x)

                loss = criterion(prediction, energy)

                val_loss += loss.item()

                pred_std.append(prediction.std().item())
                target_std.append(energy.std().item())

        val_loss /= len(val_loader)

        print(
            f"Epoch {epoch+1:02d} | "
            f"Train {train_loss:.5f} | "
            f"Val {val_loss:.5f} | "
            f"Pred std {sum(pred_std)/len(pred_std):.4f} | "
            f"Target std {sum(target_std)/len(target_std):.4f}"
        )

    torch.save(model.state_dict(), "baseline_model.pt")

    torch.save(
        {
            "energy_mean": energy_mean,
            "energy_std": energy_std,
        },
        "baseline_normalization.pt",
    )

    print("\nDone.")


if __name__ == "__main__":
    train()
