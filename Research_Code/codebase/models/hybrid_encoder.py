import torch
import torch.nn as nn


def pad_vector(x, size):
    if not hasattr(x, "__iter__") or isinstance(x, str):
        x = [x]

    x = list(x)

    if len(x) < size:
        x += [0.0] * (size - len(x))

    return x[:size]


def encode_noise(noise):
    """
    Convert noise label to numerical feature.
    """

    if noise == "depolarizing":
        return 0.0

    elif noise == "phase_damping":
        return 1.0

    else:
        return -1.0


class HybridEncoder(nn.Module):
    """
    Physics-informed hybrid encoder.

    Input:
        qubits
        depth
        noise
        gamma
        J
        h
        theta

    Output:
        latent representation z
    """

    def __init__(self, latent_dim=32):

        super().__init__()

        # Hamiltonian + circuit + noise encoder

        # qubits
        # depth
        # noise
        # gamma
        # J(3 max)
        # h(4 max assumed)

        self.classical_encoder = nn.Sequential(
            nn.Linear(11, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )

        # theta encoder

        # max theta length = 24

        self.theta_encoder = nn.Sequential(
            nn.Linear(24, 64), nn.Tanh(), nn.Linear(64, latent_dim)
        )

        # fusion

        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def dataframe_row_to_tensor(self, row):
        """
        Convert one dataframe row into model tensors.
        """

        J = pad_vector(row["J"], 3)

        h = pad_vector(row["h"], 4)

        theta = pad_vector(row["theta"], 24)

        classical = [
            float(row["qubits"]),
            float(row["depth"]),
            encode_noise(row["noise"]),
            float(row["gamma"]),
            *J,
            *h,
        ]

        return (
            torch.tensor(classical, dtype=torch.float32),
            torch.tensor(theta, dtype=torch.float32),
        )

    def forward(self, classical, theta):

        c = self.classical_encoder(classical)

        q = self.theta_encoder(theta)

        combined = torch.cat([c, q], dim=-1)

        z = self.fusion(combined)

        return z
