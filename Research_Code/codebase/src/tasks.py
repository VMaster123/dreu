import numpy as np

from qiskit.quantum_info import SparsePauliOp


class TFIMTask:

    def __init__(self, n, J_min, J_max, h_min, h_max):

        self.n = n

        self.J = np.random.uniform(J_min, J_max, n - 1)

        self.h = np.random.uniform(h_min, h_max)

    def build_hamiltonian(self):

        paulis = []

        coeffs = []

        # ZZ interactions

        for i, J in enumerate(self.J):

            label = ["I"] * self.n

            label[i] = "Z"

            label[i + 1] = "Z"

            paulis.append("".join(label[::-1]))

            coeffs.append(-J)

        # X terms

        for i in range(self.n):

            label = ["I"] * self.n

            label[i] = "X"

            paulis.append("".join(label[::-1]))

            coeffs.append(-self.h)

        return SparsePauliOp(paulis, coeffs)
