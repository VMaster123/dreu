import pandas as pd
import numpy as np
import torch


def encode_noise(noise):

    if noise == "depolarizing":
        return 0.0

    elif noise == "phase_damping":
        return 1.0

    else:
        raise ValueError(f"Unknown noise type: {noise}")


def create_features(df):

    X = []
    y = []

    for _, row in df.iterrows():

        # -------------------------
        # Hamiltonian features
        # -------------------------

        J_features = np.zeros(3)

        J_values = row["J"]

        J_features[: len(J_values)] = J_values

        h = row["h"]

        # -------------------------
        # Architecture/noise features
        # -------------------------

        features = np.concatenate(
            [
                J_features,
                [h],
                [encode_noise(row["noise"])],
                [row["gamma"]],
                [row["qubits"]],
                [row["depth"]],
            ]
        )

        X.append(features)

        # target
        y.append(row["energy"])

    X = torch.tensor(np.array(X), dtype=torch.float32)

    y = torch.tensor(np.array(y), dtype=torch.float32).reshape(-1, 1)

    return X, y


if __name__ == "__main__":

    df = pd.read_pickle("random_train.pkl")

    X, y = create_features(df)

    print("Input shape:")
    print(X.shape)

    print("Target shape:")
    print(y.shape)

    print("\nExample input:")
    print(X[0])

    print("\nExample target:")
    print(y[0])
