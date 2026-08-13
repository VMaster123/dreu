import os
import numpy as np
import pandas as pd

import config

from tasks import TFIMTask
from ansatz import create_ansatz
from simulator import expectation_value

np.random.seed(config.RANDOM_SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
)

os.makedirs(
    DATA_DIR,
    exist_ok=True,
)

DATASET_PKL = os.path.join(
    DATA_DIR,
    "dataset.pkl",
)

DATASET_CSV = os.path.join(
    DATA_DIR,
    "dataset.csv",
)


# ============================================================
# GENERATE DATASET
# ============================================================

dataset = []


for n in config.N_QUBITS:

    print(f"\nGenerating data for {n} qubits...")

    for task_id in range(config.N_TASKS):

        # ----------------------------------------------------
        # Generate Hamiltonian task
        # ----------------------------------------------------

        task = TFIMTask(
            n,
            config.J_MIN,
            config.J_MAX,
            config.H_MIN,
            config.H_MAX,
        )

        H = task.build_hamiltonian()

        # ----------------------------------------------------
        # Circuit depths
        # ----------------------------------------------------

        for depth in config.DEPTHS:

            circuit = create_ansatz(
                n,
                depth,
            )

            n_parameters = circuit.num_parameters

            # ------------------------------------------------
            # Noise models
            # ------------------------------------------------

            for noise in config.NOISE_MODELS:

                # ------------------------------------------------
                # Noise strengths
                # ------------------------------------------------

                for gamma in config.NOISE_STRENGTHS:

                    # --------------------------------------------
                    # Parameter samples
                    # --------------------------------------------

                    for _ in range(config.PARAMETER_SAMPLES):

                        # Generate parameters for this circuit.
                        #
                        # The number of parameters depends on
                        # the number of qubits and circuit depth.
                        #
                        # No fixed-size padding is used.

                        theta = np.random.uniform(
                            -np.pi,
                            np.pi,
                            n_parameters,
                        )

                        # Bind parameters to circuit

                        bound = circuit.assign_parameters(theta)

                        # Compute noisy expectation value

                        energy = expectation_value(
                            bound,
                            H,
                            noise,
                            gamma,
                        )

                        # ----------------------------------------
                        # Store sample
                        # ----------------------------------------

                        dataset.append(
                            {
                                "task_id": task_id,
                                "qubits": n,
                                "depth": depth,
                                "noise": noise,
                                "gamma": gamma,
                                # Variable-length coupling vector
                                "J": task.J.tolist(),
                                "h": float(task.h),
                                # Variable-length parameter vector
                                "theta": theta.tolist(),
                                "energy": float(energy),
                            }
                        )

    print(f"Completed {n} qubits.")


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(dataset)


# ============================================================
# SAVE DATASET
# ============================================================

df.to_pickle(DATASET_PKL)

df.to_csv(
    DATASET_CSV,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print("\n========================================")

print("DATASET GENERATION COMPLETE")

print("========================================")

print(f"Total samples: {len(df):,}")

print("\nSamples by qubit count:")

print(df["qubits"].value_counts().sort_index())

print("\nSamples by circuit depth:")

print(df["depth"].value_counts().sort_index())

print("\nSaved:")

print(f"  {DATASET_PKL}")

print(f"  {DATASET_CSV}")
