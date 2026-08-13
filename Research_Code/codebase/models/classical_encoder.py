import torch
import torch.nn as nn


def _to_float_tensor(x):
    if torch.is_tensor(x):
        return x.float()

    # Handle numeric numpy/list inputs.
    try:
        return torch.as_tensor(x, dtype=torch.float32)
    except (TypeError, ValueError):
        return torch.zeros(1, dtype=torch.float32)


def encode_noise(noise):
    """
    Robust noise encoder.

    Supports:
      - numeric tensors
      - lists / numpy arrays
      - dictionaries
      - strings / categorical noise descriptions

    Returns a fixed-size numeric tensor.
    """
    if noise is None:
        return torch.zeros(1, dtype=torch.float32)

    if torch.is_tensor(noise):
        return noise.float().flatten()

    if isinstance(noise, dict):
        values = []
        for key in sorted(noise.keys()):
            value = noise[key]

            if isinstance(value, (int, float, np_number_types())):
                values.append(float(value))
            elif isinstance(value, (list, tuple)):
                for v in value:
                    try:
                        values.append(float(v))
                    except (TypeError, ValueError):
                        continue

        if values:
            return torch.tensor(values, dtype=torch.float32)

        return torch.zeros(1, dtype=torch.float32)

    if isinstance(noise, (list, tuple)):
        values = []
        for value in noise:
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                continue

        if values:
            return torch.tensor(values, dtype=torch.float32)

        return torch.zeros(1, dtype=torch.float32)

    try:
        return torch.tensor([float(noise)], dtype=torch.float32)
    except (TypeError, ValueError):
        # Stable categorical encoding for strings.
        text = str(noise)
        encoded = [
            float(sum(ord(c) for c in text)) / 1000.0,
            float(len(text)) / 100.0,
        ]
        return torch.tensor(encoded, dtype=torch.float32)


def np_number_types():
    """
    Avoid importing numpy just for isinstance checks.
    """
    return (int, float)


class ClassicalEncoder(nn.Module):

    def __init__(
        self,
        latent_dim=64,
        max_j_positions=64,
        max_theta_positions=256,
        num_heads=4,
        classical_dim=6,
        j_feature_dim=2,
        theta_feature_dim=2,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.max_j_positions = max_j_positions
        self.max_theta_positions = max_theta_positions

        self.classical_dim = classical_dim
        self.j_feature_dim = j_feature_dim
        self.theta_feature_dim = theta_feature_dim

        # --------------------------------------------------
        # Classical input
        # --------------------------------------------------

        self.classical_net = nn.Sequential(
            nn.Linear(classical_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
        )

        # --------------------------------------------------
        # J input
        #
        # IMPORTANT:
        # J can have different lengths between train/val/test.
        #
        # We pad/truncate to max_j_positions BEFORE this network.
        # --------------------------------------------------

        self.j_net = nn.Sequential(
            nn.Linear(max_j_positions * j_feature_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
        )

        # --------------------------------------------------
        # Theta input
        # --------------------------------------------------

        self.theta_net = nn.Sequential(
            nn.Linear(max_theta_positions * theta_feature_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
        )

        # --------------------------------------------------
        # Fusion
        # --------------------------------------------------

        self.fusion = nn.Sequential(
            nn.Linear(128 + 128 + 128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(256, latent_dim),
        )

    # ======================================================
    # PAD / TRUNCATE
    # ======================================================

    @staticmethod
    def _pad_or_truncate(x, target_length):
        """
        x:
            [B, N, F]

        Returns:
            [B, target_length, F]
        """

        batch_size, length, feature_dim = x.shape

        if length > target_length:
            return x[:, :target_length, :]

        if length < target_length:
            padding = torch.zeros(
                batch_size,
                target_length - length,
                feature_dim,
                dtype=x.dtype,
                device=x.device,
            )

            return torch.cat(
                [x, padding],
                dim=1,
            )

        return x

    # ======================================================
    # FORWARD
    # ======================================================

    def forward(
        self,
        classical,
        J,
        J_mask,
        theta,
        theta_mask,
    ):

        # --------------------------------------------------
        # Classical
        # --------------------------------------------------

        classical = classical.float()

        if classical.dim() == 1:
            classical = classical.unsqueeze(0)

        # Guarantee fixed classical width.
        if classical.shape[-1] < self.classical_dim:

            padding = torch.zeros(
                classical.shape[0],
                self.classical_dim - classical.shape[-1],
                device=classical.device,
                dtype=classical.dtype,
            )

            classical = torch.cat(
                [classical, padding],
                dim=-1,
            )

        elif classical.shape[-1] > self.classical_dim:

            classical = classical[:, : self.classical_dim]

        classical_features = self.classical_net(classical)

        # --------------------------------------------------
        # J
        # --------------------------------------------------

        J = J.float()

        if J.dim() == 2:
            J = J.unsqueeze(-1)

        # If J is [B, N] treat each value as one feature.
        # If J is [B, N, 2], preserve the two features.
        if J.shape[-1] != self.j_feature_dim:

            if J.shape[-1] > self.j_feature_dim:
                J = J[..., : self.j_feature_dim]

            else:
                padding = torch.zeros(
                    *J.shape[:-1],
                    self.j_feature_dim - J.shape[-1],
                    device=J.device,
                    dtype=J.dtype,
                )

                J = torch.cat(
                    [J, padding],
                    dim=-1,
                )

        J = self._pad_or_truncate(
            J,
            self.max_j_positions,
        )

        J_flat = J.reshape(
            J.shape[0],
            self.max_j_positions * self.j_feature_dim,
        )

        j_features = self.j_net(J_flat)

        # --------------------------------------------------
        # Theta
        # --------------------------------------------------

        theta = theta.float()

        if theta.dim() == 2:
            theta = theta.unsqueeze(-1)

        if theta.shape[-1] != self.theta_feature_dim:

            if theta.shape[-1] > self.theta_feature_dim:
                theta = theta[..., : self.theta_feature_dim]

            else:
                padding = torch.zeros(
                    *theta.shape[:-1],
                    self.theta_feature_dim - theta.shape[-1],
                    device=theta.device,
                    dtype=theta.dtype,
                )

                theta = torch.cat(
                    [theta, padding],
                    dim=-1,
                )

        theta = self._pad_or_truncate(
            theta,
            self.max_theta_positions,
        )

        theta_flat = theta.reshape(
            theta.shape[0],
            self.max_theta_positions * self.theta_feature_dim,
        )

        theta_features = self.theta_net(theta_flat)

        # --------------------------------------------------
        # Fusion
        # --------------------------------------------------

        fused = torch.cat(
            [
                classical_features,
                j_features,
                theta_features,
            ],
            dim=-1,
        )

        z = self.fusion(fused)

        # Keep the latent numerically well behaved.
        z = 2.0 * torch.tanh(z / 2.0)

        return z
