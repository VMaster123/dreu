import numpy as np

import pandas as pd

import config

from tasks import TFIMTask

from ansatz import create_ansatz

from simulator import expectation_value

dataset = []

for n in config.N_QUBITS:

    for _ in range(config.N_TASKS):

        task = TFIMTask(n, config.J_MIN, config.J_MAX, config.H_MIN, config.H_MAX)

        H = task.build_hamiltonian()

        for depth in config.DEPTHS:

            circuit = create_ansatz(n, depth)

            for noise in config.NOISE_MODELS:

                for gamma in config.NOISE_STRENGTHS:

                    for _ in range(config.PARAMETER_SAMPLES):

                        theta = np.random.uniform(-np.pi, np.pi, circuit.num_parameters)

                        bound = circuit.assign_parameters(theta)

                        energy = expectation_value(
                            bound,
                            H,
                            noise,
                            gamma,
                        )

                        dataset.append(
                            {
                                "qubits": n,
                                "depth": depth,
                                "noise": noise,
                                "gamma": gamma,
                                "J": task.J.tolist(),
                                "h": task.h,
                                "theta": theta.tolist(),
                                "energy": energy,
                            }
                        )

df = pd.DataFrame(dataset)

df.to_pickle("dataset.pkl")
df.to_csv("dataset.csv", index=False)

print(df.head())
