import pandas as pd
import numpy as np

DATASET_PATH = "dataset.pkl"


def main():

    df = pd.read_pickle(DATASET_PATH)

    print("=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)

    # Basic information
    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    # Check missing values
    print("\nMissing values:")
    print(df.isnull().sum())

    # Check noise models
    print("\nNoise models:")
    print(df["noise"].value_counts())

    # Noise strengths
    print("\nGamma values:")
    print(sorted(df["gamma"].unique()))

    # Qubit counts
    print("\nQubit distribution:")
    print(df["qubits"].value_counts())

    # Circuit depths
    print("\nDepth distribution:")
    print(df["depth"].value_counts())

    # Energy statistics
    print("\nEnergy statistics:")
    print(df["energy"].describe())

    # Check J vectors
    print("\nExample J values:")
    print(df["J"].head())

    print("\nJ vector lengths:")
    print(df["J"].apply(len).value_counts())

    # Check theta lengths
    print("\nTheta vector lengths:")
    print(df["theta"].apply(len).value_counts())

    # Check duplicate rows
    print("\nDuplicate rows:")

    df_check = df.copy()

    df_check["J"] = df_check["J"].apply(tuple)
    df_check["theta"] = df_check["theta"].apply(tuple)

    print(df_check.duplicated().sum())

    # Physics sanity check:
    # energy should be finite
    print("\nFinite energies:")
    print(np.isfinite(df["energy"]).all())

    # Compare noise effects
    print("\nAverage energy by noise:")
    print(df.groupby(["noise", "gamma"])["energy"].mean())

    print("\nEnergy by noise strength:")
    print(df.groupby(["noise", "gamma"])["energy"].agg(["mean", "std"]))


if __name__ == "__main__":
    main()
