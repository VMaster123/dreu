from tasks import TFIMTask
from ansatz import create_ansatz
from simulator import expectation_value
import numpy as np

task = TFIMTask(
    n=3,
    J_min=0.5,
    J_max=1.5,
    h_min=0.5,
    h_max=1.5,
)

H = task.build_hamiltonian()

circuit = create_ansatz(3, 1)

theta = np.random.uniform(
    -np.pi,
    np.pi,
    circuit.num_parameters,
)

bound = circuit.assign_parameters(theta)

print("Ideal:")
print(expectation_value(bound, H))

print("Depolarizing:")
print(expectation_value(bound, H, "depolarizing", 0.8))

print("Phase Damping:")
print(expectation_value(bound, H, "phase_damping", 0.8))
