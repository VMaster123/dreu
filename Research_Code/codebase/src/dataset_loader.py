import os
import sys

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

# ============================================================
# PATHS / IMPORTS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(
    0,
    BASE_DIR,
)

from models.classical_encoder import encode_noise

# ============================================================
# DATASET
# ============================================================


class QuantumDataset(Dataset):

    def __init__(
        self,
        dataset_path,
        energy_mean=None,
        energy_std=None,
        input_mean=None,
        input_std=None,
    ):

        self.dataset_path = dataset_path

        self.df = pd.read_pickle(dataset_path)

        if len(self.df) == 0:
            raise ValueError(f"Dataset is empty: {dataset_path}")

        required_columns = {
            "qubits",
            "depth",
            "gamma",
            "noise",
            "J",
            "h",
            "theta",
            "energy",
        }

        missing = required_columns - set(self.df.columns)

        if missing:
            raise ValueError(
                "Dataset is missing required columns: " + ", ".join(sorted(missing))
            )

        # ====================================================
        # GLOBAL FEATURES
        # ====================================================

        classical_features = []

        for _, row in self.df.iterrows():

            features = [
                float(row["qubits"]),
                float(row["depth"]),
                float(row["gamma"]),
                *encode_noise(row["noise"]),
                float(row["h"]),
            ]

            classical_features.append(features)

        classical_features = np.asarray(
            classical_features,
            dtype=np.float32,
        )

        # ====================================================
        # NORMALIZATION
        # ====================================================

        if input_mean is None:

            self.input_mean = classical_features.mean(axis=0)

            self.input_std = classical_features.std(axis=0)

        else:

            self.input_mean = np.asarray(
                input_mean,
                dtype=np.float32,
            )

            self.input_std = np.asarray(
                input_std,
                dtype=np.float32,
            )

        invalid = ~np.isfinite(self.input_std) | (self.input_std < 1e-8)

        self.input_std[invalid] = 1.0

        # ====================================================
        # ENERGY NORMALIZATION
        # ====================================================

        if energy_mean is None:

            self.energy_mean = float(self.df["energy"].mean())

        else:

            self.energy_mean = float(energy_mean)

        if energy_std is None:

            self.energy_std = float(self.df["energy"].std())

        else:

            self.energy_std = float(energy_std)

        if not np.isfinite(self.energy_std) or self.energy_std < 1e-8:
            self.energy_std = 1.0

        # ====================================================
        # INFORMATION
        # ====================================================

        self.classical_dim = 6

        self.min_qubits = int(self.df["qubits"].min())

        self.max_qubits = int(self.df["qubits"].max())

        self.max_j_dim = max(len(np.asarray(j)) for j in self.df["J"])

        self.max_theta_dim = max(len(np.asarray(theta)) for theta in self.df["theta"])

        print("\nQuantumDataset loaded:")

        print(f"  Samples       : {len(self.df):,}")

        print(f"  Qubits        : " f"{self.min_qubits} -> " f"{self.max_qubits}")

        print(f"  Max J length  : " f"{self.max_j_dim}")

        print(f"  Max theta     : " f"{self.max_theta_dim}")

        print(f"  Classical dim : " f"{self.classical_dim}")

        print(f"  Energy mean   : " f"{self.energy_mean:.6f}")

        print(f"  Energy std    : " f"{self.energy_std:.6f}")

    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):

        return len(self.df)

    # ========================================================
    # SAMPLE
    # ========================================================

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        # ====================================================
        # GLOBAL FEATURES
        # ====================================================

        classical = np.asarray(
            [
                float(row["qubits"]),
                float(row["depth"]),
                float(row["gamma"]),
                *encode_noise(row["noise"]),
                float(row["h"]),
            ],
            dtype=np.float32,
        )

        classical = (classical - self.input_mean) / self.input_std

        # ====================================================
        # J
        # ====================================================

        J = np.asarray(
            row["J"],
            dtype=np.float32,
        ).reshape(-1)

        # ====================================================
        # THETA
        #
        # Keep every angle.
        #
        # sin/cos avoids the discontinuity at +/- pi.
        # ====================================================

        theta = np.asarray(
            row["theta"],
            dtype=np.float32,
        ).reshape(-1)

        theta_features = np.stack(
            [
                np.sin(theta),
                np.cos(theta),
            ],
            axis=-1,
        ).astype(np.float32)

        # ====================================================
        # ENERGY
        # ====================================================

        energy = (float(row["energy"]) - self.energy_mean) / self.energy_std

        return (
            torch.tensor(
                classical,
                dtype=torch.float32,
            ),
            torch.tensor(
                J,
                dtype=torch.float32,
            ),
            torch.tensor(
                theta_features,
                dtype=torch.float32,
            ),
            torch.tensor(
                [energy],
                dtype=torch.float32,
            ),
        )


# ============================================================
# COLLATOR
# ============================================================


def quantum_collate_fn(batch):

    classical_list = []
    J_list = []
    theta_list = []
    energy_list = []

    for (
        classical,
        J,
        theta,
        energy,
    ) in batch:

        classical_list.append(classical)
        J_list.append(J)
        theta_list.append(theta)
        energy_list.append(energy)

    # ========================================================
    # GLOBAL
    # ========================================================

    classical = torch.stack(classical_list)

    # ========================================================
    # J
    # ========================================================

    batch_size = len(J_list)

    max_j = max(x.shape[0] for x in J_list)

    J_padded = torch.zeros(
        batch_size,
        max_j,
        dtype=torch.float32,
    )

    J_mask = torch.zeros(
        batch_size,
        max_j,
        dtype=torch.bool,
    )

    for i, J in enumerate(J_list):

        length = J.shape[0]

        J_padded[i, :length] = J

        J_mask[i, :length] = True

    # ========================================================
    # THETA
    # ========================================================

    max_theta = max(x.shape[0] for x in theta_list)

    theta_padded = torch.zeros(
        batch_size,
        max_theta,
        2,
        dtype=torch.float32,
    )

    theta_mask = torch.zeros(
        batch_size,
        max_theta,
        dtype=torch.bool,
    )

    for i, theta in enumerate(theta_list):

        length = theta.shape[0]

        theta_padded[i, :length, :] = theta

        theta_mask[i, :length] = True

    # ========================================================
    # ENERGY
    # ========================================================

    energy = torch.stack(energy_list)

    return {
        "classical": classical,
        "J": J_padded,
        "J_mask": J_mask,
        "theta": theta_padded,
        "theta_mask": theta_mask,
        "energy": energy,
    }
