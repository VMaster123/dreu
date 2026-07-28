from qiskit.circuit.library import EfficientSU2


def create_ansatz(n_qubits, depth):

    return EfficientSU2(num_qubits=n_qubits, reps=depth, entanglement="linear")
