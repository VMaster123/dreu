"""
noise.py

Implements the quantum noise channels used in the thesis.

Supported channels
------------------
1. Global Depolarizing
2. Local Phase Damping

Author: Vilohith Gokarakonda
"""

import numpy as np
from qiskit.quantum_info import DensityMatrix


def kron_n(operators):
    """
    Compute the Kronecker product of a list of matrices.

    Example
    -------
    kron_n([A, B, C]) = A tensor B  tensor C
    """

    result = operators[0]

    for op in operators[1:]:
        result = np.kron(result, op)

    return result


def depolarizing_channel(rho: DensityMatrix, p: float) -> DensityMatrix:
    """
    Apply global depolarizing noise.

    rho' = (1-p) rho + p I/d
    """

    if p == 0:
        return rho

    d = rho.dim

    identity = np.eye(d, dtype=complex) / d

    noisy = (1 - p) * rho.data + p * identity

    return DensityMatrix(noisy)


def phase_damping_channel(
    rho: DensityMatrix,
    gamma: float,
) -> DensityMatrix:
    """
    Apply independent phase damping
    to every qubit.

    Kraus operators

        K0 = [[1,0],
              [0,sqrt(1-gamma)]]

        K1 = [[0,0],
              [0,sqrt(gamma)]]
    """

    if gamma == 0:
        return rho

    n = rho.num_qubits

    I = np.eye(2, dtype=complex)

    K0 = np.array(
        [
            [1, 0],
            [0, np.sqrt(1 - gamma)],
        ],
        dtype=complex,
    )

    K1 = np.array(
        [
            [0, 0],
            [0, np.sqrt(gamma)],
        ],
        dtype=complex,
    )

    current = rho.data.copy()

    #
    # Apply the channel to each qubit
    #

    for target in range(n):

        ops0 = []
        ops1 = []

        for q in range(n):

            if q == target:

                ops0.append(K0)
                ops1.append(K1)

            else:

                ops0.append(I)
                ops1.append(I)

        K0_full = kron_n(ops0)
        K1_full = kron_n(ops1)

        current = (
            K0_full @ current @ K0_full.conj().T + K1_full @ current @ K1_full.conj().T
        )

    return DensityMatrix(current)


def apply_noise(
    rho: DensityMatrix,
    noise_type: str,
    gamma: float,
) -> DensityMatrix:
    """
    Dispatch function.
    """

    if noise_type is None:
        return rho

    if gamma == 0:
        return rho

    if noise_type == "depolarizing":

        return depolarizing_channel(
            rho,
            gamma,
        )

    if noise_type == "phase_damping":

        return phase_damping_channel(
            rho,
            gamma,
        )

    raise ValueError(f"Unknown noise model '{noise_type}'")
