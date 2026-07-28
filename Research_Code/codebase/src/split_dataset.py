import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_PATH = "dataset.pkl"


def save_split(train, val, test, name):

    train.to_pickle(f"{name}_train.pkl")
    val.to_pickle(f"{name}_val.pkl")
    test.to_pickle(f"{name}_test.pkl")

    print(f"\n{name} split:")
    print("train:", train.shape)
    print("val:  ", val.shape)
    print("test: ", test.shape)


def random_split(df):

    train, temp = train_test_split(df, test_size=0.2, random_state=42)

    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    save_split(train, val, test, "random")


def noise_transfer_split(df):

    # Train only depolarizing
    train = df[df["noise"] == "depolarizing"]

    # Test phase damping
    test = df[df["noise"] == "phase_damping"]

    train, val = train_test_split(train, test_size=0.2, random_state=42)

    save_split(train, val, test, "noise_transfer")


def architecture_transfer_split(df):

    # Train on 3 qubits
    train = df[df["qubits"] == 3]

    # Test on 4 qubits
    test = df[df["qubits"] == 4]

    train, val = train_test_split(train, test_size=0.2, random_state=42)

    save_split(train, val, test, "architecture_transfer")


def main():

    df = pd.read_pickle(DATASET_PATH)

    print("Dataset:")
    print(df.shape)

    random_split(df)

    noise_transfer_split(df)

    architecture_transfer_split(df)


if __name__ == "__main__":
    main()
