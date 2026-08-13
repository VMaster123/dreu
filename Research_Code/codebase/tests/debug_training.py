import sys
import os

sys.path.append(os.path.abspath(".."))

import torch
import pandas as pd
from torch.utils.data import DataLoader

from src.dataset_loader import QuantumDataset
from models.hybrid_encoder import HybridEncoder
from models.surrogate_nn import SurrogateModel

DATA_PATH = "../data/random_train.pkl"


from src.dataset_loader import QuantumDataset


def main():

    df = pd.read_pickle("../data/random_train.pkl")

    print(df.columns)
    print(df.iloc[0])

    print("=" * 60)
    print("TRAINING DEBUG")
    print("=" * 60)

    dataset = QuantumDataset(DATA_PATH)

    print("\nDataset size:", len(dataset))

    classical, theta, energy = dataset[0]

    print("\nSingle sample")
    print("----------------")
    print("Classical shape:", classical.shape)
    print("Theta shape:", theta.shape)
    print("Energy:", energy)

    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    classical, theta, energy = next(iter(loader))

    print("\nBatch")
    print("----------------")
    print("Classical:", classical.shape)
    print("Theta:", theta.shape)
    print("Energy:", energy.shape)

    encoder = HybridEncoder()
    surrogate = SurrogateModel()

    z = encoder(classical, theta)

    print("\nEncoder output")
    print("----------------")
    print("Latent shape:", z.shape)

    prediction = surrogate(z)

    print("\nPredictions")
    print("----------------")
    print(prediction)

    print("\nTargets")
    print("----------------")
    print(energy)

    criterion = torch.nn.MSELoss()

    loss = criterion(prediction, energy)

    print("\nInitial loss:", loss.item())

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(surrogate.parameters()), lr=1e-3
    )

    optimizer.zero_grad()
    loss.backward()

    print("\nGradient norms")
    print("----------------")

    for name, param in encoder.named_parameters():

        if param.grad is None:
            print(name, "None")
        else:
            print(name, param.grad.norm().item())

    for name, param in surrogate.named_parameters():

        if param.grad is None:
            print(name, "None")
        else:
            print(name, param.grad.norm().item())

    optimizer.step()

    new_prediction = surrogate(encoder(classical, theta)).squeeze()

    new_loss = criterion(new_prediction, energy)

    print("\nLoss after ONE optimizer step")
    print("----------------")
    print("Old:", loss.item())
    print("New:", new_loss.item())

    print("\nFirst five predictions")
    print(prediction[:5])

    print("\nFirst five new predictions")
    print(new_prediction[:5])

    print("\nFirst five targets")
    print(energy[:5])

    print("NEW STUFFF")
    df = pd.read_pickle("../data/random_train.pkl")

    print(df["energy"].std())
    print(df["energy"].mean())

    for name, param in surrogate.named_parameters():
        print(name, param.data.norm().item())

    print("NEW STUFFF")

    dataset = QuantumDataset("../data/random_train.pkl")

    for i in range(5):
        classical, theta, energy = dataset[i]
        print(classical)
        print(theta[:5])
        print(energy)
        print()


if __name__ == "__main__":
    main()
