import sys
import os

sys.path.append(os.path.abspath(".."))

import torch

from models.surrogate_nn import SurrogateModel


def test_surrogate():

    surrogate = SurrogateModel()

    # fake latent vector from encoder
    z = torch.randn(1, 32)

    prediction = surrogate(z)

    print("Latent shape:", z.shape)
    print("Prediction shape:", prediction.shape)
    print("Prediction:", prediction)


if __name__ == "__main__":
    test_surrogate()
