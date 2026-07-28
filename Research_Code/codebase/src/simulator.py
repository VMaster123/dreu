from qiskit.quantum_info import DensityMatrix

from noise import apply_noise


def expectation_value(
    circuit,
    hamiltonian,
    noise_type=None,
    gamma=0.0,
):
    """
    Compute

        Tr(H rho)

    after applying the selected
    quantum noise channel.
    """

    rho = DensityMatrix.from_instruction(circuit)

    rho = apply_noise(
        rho,
        noise_type,
        gamma,
    )

    energy = rho.expectation_value(hamiltonian)

    return energy.real
