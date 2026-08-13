from dataset_loader import QuantumDataset, quantum_collate_fn
from models.classical_encoder import ClassicalEncoder
from torch.utils.data import DataLoader

dataset = QuantumDataset("data/hamiltonian_train.pkl")

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=quantum_collate_fn,
)

batch = next(iter(loader))

print("classical:", batch["classical"].shape)
print("J:", batch["J"].shape)
print("J_mask:", batch["J_mask"].shape)
print("theta:", batch["theta"].shape)
print("theta_mask:", batch["theta_mask"].shape)
print("energy:", batch["energy"].shape)

encoder = ClassicalEncoder(latent_dim=64)

z = encoder(
    batch["classical"],
    batch["J"],
    batch["J_mask"],
    batch["theta"],
    batch["theta_mask"],
)

print("latent:", z.shape)
