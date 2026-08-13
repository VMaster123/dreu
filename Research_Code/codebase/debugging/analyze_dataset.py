import pandas as pd
import numpy as np

df = pd.read_pickle("dataset.pkl")

print("Total samples:", len(df))

print("\nQubit counts:")
print(df["qubits"].value_counts().sort_index())

print("\nJ lengths:")
print(df.groupby("qubits")["J"].apply(lambda x: sorted(set(len(j) for j in x))))

print("\nTheta lengths:")
print(
    df.groupby(["qubits", "depth"])["theta"].apply(
        lambda x: sorted(set(len(t) for t in x))
    )
)

print("\nEnergy validity:")
print("NaN energies:", df["energy"].isna().sum())
print("Infinite energies:", np.isinf(df["energy"]).sum())

print("\nExample rows:")

for n in sorted(df["qubits"].unique()):
    row = df[df["qubits"] == n].iloc[0]

    print(
        f"{n} qubits | "
        f"J length = {len(row['J'])} | "
        f"theta length = {len(row['theta'])} | "
        f"energy = {row['energy']}"
    )
