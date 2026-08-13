import os

import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "data",
    "dataset.pkl",
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
)

RANDOM_STATE = 42


# ============================================================
# SAVE SPLIT
# ============================================================


def save_split(
    train,
    val,
    test,
    name,
):
    """
    Save train/validation/test datasets for one
    transfer experiment.
    """

    train = train.copy()
    val = val.copy()
    test = test.copy()

    for split in (train, val, test):
        if "ham_id" in split.columns:
            split.drop(
                columns=["ham_id"],
                inplace=True,
            )

    train_path = os.path.join(
        OUTPUT_DIR,
        f"{name}_train.pkl",
    )

    val_path = os.path.join(
        OUTPUT_DIR,
        f"{name}_val.pkl",
    )

    test_path = os.path.join(
        OUTPUT_DIR,
        f"{name}_test.pkl",
    )

    train.to_pickle(train_path)
    val.to_pickle(val_path)
    test.to_pickle(test_path)

    print(f"\n{name} split:")
    print("  train:", train.shape)
    print("  val:  ", val.shape)
    print("  test: ", test.shape)

    print("\nSaved:")
    print(" ", train_path)
    print(" ", val_path)
    print(" ", test_path)


# ============================================================
# HAMILTONIAN TRANSFER
# ============================================================


def hamiltonian_split(df):
    """
    Train/validation and test contain completely different
    Hamiltonians.

    No Hamiltonian appears in more than one split.
    """

    df = df.copy()

    def make_hamiltonian_id(row):
        J = tuple(float(x) for x in row["J"])

        h_value = row["h"]

        if hasattr(h_value, "__iter__") and not isinstance(
            h_value,
            (str, bytes),
        ):
            h = tuple(float(x) for x in h_value)
        else:
            h = float(h_value)

        return (
            int(row["qubits"]),
            J,
            h,
        )

    df["ham_id"] = df.apply(
        make_hamiltonian_id,
        axis=1,
    )

    hamiltonians = df["ham_id"].unique()

    train_h, temp_h = train_test_split(
        hamiltonians,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    val_h, test_h = train_test_split(
        temp_h,
        test_size=0.50,
        random_state=RANDOM_STATE,
    )

    train = df[df["ham_id"].isin(train_h)]

    val = df[df["ham_id"].isin(val_h)]

    test = df[df["ham_id"].isin(test_h)]

    save_split(
        train,
        val,
        test,
        "hamiltonian",
    )


# ============================================================
# NOISE TRANSFER
# ============================================================


def noise_transfer_split(df):
    """
    Train on depolarizing noise.

    Test on phase-damping noise.

    This deliberately tests whether information learned under
    one noise model transfers to a different noise model.
    """

    df = df.copy()

    train_pool = df[df["noise"] == "depolarizing"].copy()

    test = df[df["noise"] == "phase_damping"].copy()

    if train_pool.empty:
        raise ValueError("No depolarizing samples found.")

    if test.empty:
        raise ValueError("No phase_damping samples found.")

    train, val = train_test_split(
        train_pool,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    save_split(
        train,
        val,
        test,
        "noise_transfer",
    )


# ============================================================
# ARCHITECTURE TRANSFER
# ============================================================


def architecture_transfer_split(df):
    """
    Train on smaller quantum systems and test on larger,
    completely unseen systems.

    Dataset contains 3–8 qubit systems.

    Training:
        3, 4, 5, 6 qubits

    Validation:
        held-out samples from 3–6 qubits

    Test:
        7, 8 qubits

    The 7- and 8-qubit systems are never seen during training.
    """

    df = df.copy()

    train_pool = df[df["qubits"].isin([3, 4, 5, 6])].copy()

    test = df[df["qubits"].isin([7, 8])].copy()

    if train_pool.empty:
        raise ValueError("No 3–6 qubit samples found.")

    if test.empty:
        raise ValueError("No 7–8 qubit samples found.")

    train, val = train_test_split(
        train_pool,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    save_split(
        train,
        val,
        test,
        "architecture_transfer",
    )


# ============================================================
# MAIN
# ============================================================


def main():

    print("\n========================================")
    print("CREATING TRANSFER BENCHMARKS")
    print("========================================")

    print("\nDataset:")
    print(DATASET_PATH)

    df = pd.read_pickle(DATASET_PATH)

    print("\nDataset shape:")
    print(df.shape)

    print("\nQubit counts:")

    print(df["qubits"].value_counts().sort_index())

    required_columns = {
        "qubits",
        "J",
        "h",
        "theta",
        "gamma",
        "noise",
        "energy",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(sorted(missing))
        )

    # ========================================================
    # HAMILTONIAN TRANSFER
    # ========================================================

    print("\n----------------------------------------")
    print("HAMILTONIAN TRANSFER")
    print("----------------------------------------")

    hamiltonian_split(df)

    # ========================================================
    # NOISE TRANSFER
    # ========================================================

    print("\n----------------------------------------")
    print("NOISE-MODEL TRANSFER")
    print("----------------------------------------")

    noise_transfer_split(df)

    # ========================================================
    # ARCHITECTURE TRANSFER
    # ========================================================

    print("\n----------------------------------------")
    print("ARCHITECTURE TRANSFER")
    print("----------------------------------------")

    architecture_transfer_split(df)

    print("\n========================================")
    print("ALL TRANSFER SPLITS CREATED")
    print("========================================")


if __name__ == "__main__":
    main()
