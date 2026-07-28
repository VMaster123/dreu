import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hybrid_encoder import HybridEncoder


def test_encoder():

    df = pd.read_pickle("../data/dataset.pkl")

    row = df.iloc[0]

    encoder = HybridEncoder()

    classical, theta = encoder.dataframe_row_to_tensor(row)

    print("Classical features:")
    print(classical)
    print("Classical length:", len(classical))

    print("\nTheta features:")
    print(theta)
    print("Theta length:", len(theta))

    # add batch dimension
    classical = classical.unsqueeze(0)
    theta = theta.unsqueeze(0)

    z = encoder(classical, theta)

    print("\nLatent vector:")
    print(z)

    print("Latent shape:")
    print(z.shape)


if __name__ == "__main__":
    test_encoder()
