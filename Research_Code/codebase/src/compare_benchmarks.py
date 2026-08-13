import os
import sys
import json
import glob
import random

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset

# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed()


# ============================================================
# PATHS
# ============================================================

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


from dataset_loader import (
    QuantumDataset,
    quantum_collate_fn,
)

from models.classical_encoder import ClassicalEncoder

# ============================================================
# DIRECTORIES
# ============================================================

DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 256
EPOCHS = 100
LR = 3e-4
PATIENCE = 15
LATENT_DIM = 64
WEIGHT_DECAY = 1e-4


# ============================================================
# DEBUG
# ============================================================

# KEEP THIS TRUE UNTIL THE MODEL PASSES THE OVERFIT TEST.
DEBUG_MODE = False

DEBUG_TRAIN_SAMPLES = 2048
DEBUG_VAL_SAMPLES = 2048

# Explicit tiny overfit test.
RUN_OVERFIT_TEST = True

OVERFIT_SAMPLES = 128
OVERFIT_EPOCHS = 300
OVERFIT_LR = 2e-4


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\nUsing device:", DEVICE)


# ============================================================
# SURROGATE
# ============================================================


class SurrogateModel(nn.Module):
    """
    Stronger surrogate than a single shallow head.

    Input:
        latent representation z

    Output:
        normalized energy
    """

    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, z):
        return self.net(z)


# ============================================================
# BENCHMARK DISCOVERY
# ============================================================


def discover_benchmarks():

    train_pattern = os.path.join(
        DATA_DIR,
        "*_train.pkl",
    )

    train_files = sorted(glob.glob(train_pattern))

    print("\nFound training files:")

    if not train_files:
        raise FileNotFoundError(f"No training files found in:\n{DATA_DIR}")

    for path in train_files:
        print("  ", os.path.basename(path))

    benchmarks = {}

    for train_path in train_files:

        filename = os.path.basename(train_path)

        if not filename.endswith("_train.pkl"):
            continue

        benchmark = filename[: -len("_train.pkl")]

        val_path = os.path.join(
            DATA_DIR,
            f"{benchmark}_val.pkl",
        )

        test_path = os.path.join(
            DATA_DIR,
            f"{benchmark}_test.pkl",
        )

        print(f"\nChecking benchmark: {benchmark}")

        print(
            "Train:",
            os.path.exists(train_path),
            train_path,
        )

        print(
            "Val:  ",
            os.path.exists(val_path),
            val_path,
        )

        print(
            "Test: ",
            os.path.exists(test_path),
            test_path,
        )

        if not os.path.exists(val_path):
            print(f"Skipping {benchmark}: " "validation file not found.")
            continue

        if not os.path.exists(test_path):
            print(f"Skipping {benchmark}: " "test file not found.")
            continue

        benchmarks[benchmark] = {
            "train": train_path,
            "val": val_path,
            "test": test_path,
        }

    if not benchmarks:
        raise FileNotFoundError("No complete benchmark datasets were found.")

    print("\nDiscovered benchmarks:")

    for name in benchmarks:
        print(" -", name)

    return benchmarks


# ============================================================
# MODEL
# ============================================================


def create_encoder():
    return ClassicalEncoder(
        latent_dim=LATENT_DIM,
        max_j_positions=64,
        max_theta_positions=256,
        num_heads=4,
    ).to(DEVICE)


def create_surrogate():
    return SurrogateModel(input_dim=LATENT_DIM).to(DEVICE)


# ============================================================
# DATA
# ============================================================


def load_data(
    train_path,
    val_path,
    test_path,
):

    print("\nLoading datasets...")

    train_dataset = QuantumDataset(train_path)

    energy_mean = train_dataset.energy_mean
    energy_std = train_dataset.energy_std

    input_mean = train_dataset.input_mean
    input_std = train_dataset.input_std

    val_dataset = QuantumDataset(
        val_path,
        energy_mean=energy_mean,
        energy_std=energy_std,
        input_mean=input_mean,
        input_std=input_std,
    )

    test_dataset = QuantumDataset(
        test_path,
        energy_mean=energy_mean,
        energy_std=energy_std,
        input_mean=input_mean,
        input_std=input_std,
    )

    print("\nDataset sizes:")

    print(
        "Train:",
        len(train_dataset),
    )

    print(
        "Validation:",
        len(val_dataset),
    )

    print(
        "Test:",
        len(test_dataset),
    )

    # --------------------------------------------------------
    # DEBUG SUBSETS
    # --------------------------------------------------------

    if DEBUG_MODE:

        train_count = min(
            DEBUG_TRAIN_SAMPLES,
            len(train_dataset),
        )

        val_count = min(
            DEBUG_VAL_SAMPLES,
            len(val_dataset),
        )

        rng_train = np.random.default_rng(SEED)

        rng_val = np.random.default_rng(SEED + 1)

        train_indices = rng_train.choice(
            len(train_dataset),
            size=train_count,
            replace=False,
        )

        val_indices = rng_val.choice(
            len(val_dataset),
            size=val_count,
            replace=False,
        )

        train_dataset = Subset(
            train_dataset,
            train_indices.tolist(),
        )

        val_dataset = Subset(
            val_dataset,
            val_indices.tolist(),
        )

        print("\nDEBUG MODE")

        print(
            "Training samples:",
            len(train_dataset),
        )

        print(
            "Validation samples:",
            len(val_dataset),
        )

    # --------------------------------------------------------
    # LOADERS
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=quantum_collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=quantum_collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=quantum_collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        energy_mean,
        energy_std,
        input_mean,
        input_std,
    )


# ============================================================
# DEVICE TRANSFER
# ============================================================


def move_batch_to_device(batch):

    return {
        key: (
            value.to(
                DEVICE,
                non_blocking=True,
            )
            if torch.is_tensor(value)
            else value
        )
        for key, value in batch.items()
    }


# ============================================================
# FORWARD
# ============================================================


def forward_encoder(
    encoder,
    surrogate,
    batch,
):
    z = encoder(
        batch["classical"],
        batch["J"],
        batch["J_mask"],
        batch["theta"],
        batch["theta_mask"],
    )

    prediction = surrogate(z)

    return prediction, z


# ============================================================
# OVERFIT TEST
# ============================================================


def run_overfit_test(train_loader):
    """
    Strict sanity test.

    Loads exactly 128 examples and repeatedly trains on those
    same examples. The purpose is ONLY to verify that the model
    and data interface have enough capacity to memorize data.
    """

    print("\n" + "=" * 60)
    print("OVERFIT SANITY TEST")
    print("=" * 60)
    print("Goal: force the model to memorize 128 samples.")

    # --------------------------------------------------------
    # Collect exactly one fixed dataset
    # --------------------------------------------------------

    batches = []

    total = 0

    for batch in train_loader:

        batch_size = batch["energy"].shape[0]

        remaining = 128 - total

        if remaining <= 0:
            break

        if batch_size > remaining:
            batch = {k: v[:remaining] for k, v in batch.items()}

        batches.append(batch)

        total += batch["energy"].shape[0]

        if total >= 128:
            break

    if total < 128:
        raise RuntimeError(f"Could only collect {total} samples for overfit test.")

    # Concatenate fixed tensors.

    fixed_batch = {}

    for key in batches[0].keys():

        values = [b[key] for b in batches]

        try:
            fixed_batch[key] = torch.cat(values, dim=0)
        except RuntimeError:
            fixed_batch[key] = values[0]

    fixed_batch = {k: v.to(DEVICE) for k, v in fixed_batch.items()}

    print(
        "Actual overfit samples:",
        fixed_batch["energy"].shape[0],
    )

    # --------------------------------------------------------
    # Fresh model
    # --------------------------------------------------------

    encoder = create_encoder()
    surrogate = create_surrogate()

    encoder.train()
    surrogate.train()

    parameters = list(encoder.parameters()) + list(surrogate.parameters())

    optimizer = torch.optim.Adam(
        parameters,
        lr=3e-3,
        weight_decay=0.0,
    )

    criterion = nn.MSELoss()

    # --------------------------------------------------------
    # Initial loss
    # --------------------------------------------------------

    with torch.no_grad():

        prediction, _ = forward_encoder(
            encoder,
            surrogate,
            fixed_batch,
        )

        initial_loss = criterion(
            prediction,
            fixed_batch["energy"],
        ).item()

    # --------------------------------------------------------
    # Train repeatedly on EXACT SAME batch
    # --------------------------------------------------------

    final_loss = initial_loss

    for epoch in range(1000):

        encoder.train()
        surrogate.train()

        optimizer.zero_grad(set_to_none=True)

        prediction, z = forward_encoder(
            encoder,
            surrogate,
            fixed_batch,
        )

        loss = criterion(
            prediction,
            fixed_batch["energy"],
        )

        loss.backward()

        optimizer.step()

        final_loss = loss.item()

        if epoch == 0 or (epoch + 1) % 55 == 0:
            print(f"Overfit Epoch {epoch + 1:03d}/1000 | " f"MSE = {final_loss:.8f}")

        if final_loss < 2e-3:
            print(f"\nOverfit converged at epoch {epoch + 1}.")
            break

    print()

    if final_loss < 2e-3:

        print("OVERFIT TEST PASSED.")
        print(f"Initial MSE: {initial_loss:.8f}")
        print(f"Final MSE:   {final_loss:.8f}")

        return True

    print("OVERFIT TEST FAILED.")
    print(f"Initial MSE: {initial_loss:.8f}")
    print(f"Final MSE:   {final_loss:.8f}")

    return False


# ============================================================
# TRAIN
# ============================================================


def train_model(
    train_loader,
    val_loader,
    benchmark,
):

    encoder = create_encoder()

    surrogate = create_surrogate()

    parameters = list(encoder.parameters()) + list(surrogate.parameters())

    optimizer = torch.optim.AdamW(
        parameters,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.MSELoss()

    best_val = float("inf")

    best_encoder = None
    best_surrogate = None

    train_history = []
    val_history = []

    patience = 0

    print("\n========================================")
    print("TRAINING:", benchmark)
    print("========================================")

    for epoch in range(EPOCHS):

        # ====================================================
        # TRAIN
        # ====================================================

        encoder.train()
        surrogate.train()

        total_loss = 0.0
        total_samples = 0

        for batch in train_loader:

            batch = move_batch_to_device(batch)

            optimizer.zero_grad(set_to_none=True)

            prediction, z = forward_encoder(
                encoder,
                surrogate,
                batch,
            )

            loss = criterion(
                prediction,
                batch["energy"],
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                parameters,
                max_norm=5.0,
            )

            optimizer.step()

            n = batch["energy"].shape[0]

            total_loss += loss.item() * n

            total_samples += n

        train_loss = total_loss / max(total_samples, 1)

        # ====================================================
        # VALIDATION
        # ====================================================

        encoder.eval()
        surrogate.eval()

        total_loss = 0.0
        total_samples = 0

        latent_norms = []

        with torch.no_grad():

            for batch in val_loader:

                batch = move_batch_to_device(batch)

                prediction, z = forward_encoder(
                    encoder,
                    surrogate,
                    batch,
                )

                loss = criterion(
                    prediction,
                    batch["energy"],
                )

                n = batch["energy"].shape[0]

                total_loss += loss.item() * n

                total_samples += n

                latent_norms.append(z.norm(dim=1).mean().item())

        val_loss = total_loss / max(total_samples, 1)

        latent_norm = float(np.mean(latent_norms))

        train_history.append(train_loss)

        val_history.append(val_loss)

        # ====================================================
        # CHECKPOINT
        # ====================================================

        improved = val_loss < best_val

        if improved:

            best_val = val_loss

            best_encoder = {
                key: value.detach().cpu().clone()
                for key, value in encoder.state_dict().items()
            }

            best_surrogate = {
                key: value.detach().cpu().clone()
                for key, value in surrogate.state_dict().items()
            }

            patience = 0

        else:

            patience += 1

        marker = "*" if improved else ""

        print(
            f"Epoch {epoch + 1:03d}/{EPOCHS} | "
            f"Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f} | "
            f"Latent norm: {latent_norm:.4f} | "
            f"Patience: "
            f"{patience}/{PATIENCE} "
            f"{marker}"
        )

        if patience >= PATIENCE:

            print("\nEarly stopping.")

            break

    # ========================================================
    # RESTORE BEST
    # ========================================================

    if best_encoder is None:

        raise RuntimeError("Training never produced a valid checkpoint.")

    encoder.load_state_dict(best_encoder)

    surrogate.load_state_dict(best_surrogate)

    return (
        encoder,
        surrogate,
        train_history,
        val_history,
        best_val,
    )


# ============================================================
# TEST
# ============================================================


def evaluate(
    encoder,
    surrogate,
    test_loader,
    energy_mean,
    energy_std,
):

    encoder.eval()
    surrogate.eval()

    predictions = []
    targets = []

    with torch.no_grad():

        for batch in test_loader:

            batch = move_batch_to_device(batch)

            prediction, _ = forward_encoder(
                encoder,
                surrogate,
                batch,
            )

            predictions.append(prediction.cpu())

            targets.append(batch["energy"].cpu())

    predictions = torch.cat(predictions)

    targets = torch.cat(targets)

    mse = torch.mean((predictions - targets) ** 2).item()

    rmse = float(np.sqrt(mse))

    # --------------------------------------------------------
    # Original energy units
    # --------------------------------------------------------

    if torch.is_tensor(energy_mean):

        mean = energy_mean.cpu()

    else:

        mean = torch.tensor(energy_mean)

    if torch.is_tensor(energy_std):

        std = energy_std.cpu()

    else:

        std = torch.tensor(energy_std)

    prediction_original = predictions * std + mean

    target_original = targets * std + mean

    original_mse = torch.mean((prediction_original - target_original) ** 2).item()

    original_rmse = float(np.sqrt(original_mse))

    return {
        "mse": mse,
        "rmse": rmse,
        "mse_original": original_mse,
        "rmse_original": original_rmse,
    }


# ============================================================
# SAVE
# ============================================================


def save_results(
    benchmark,
    encoder,
    surrogate,
    train_history,
    val_history,
    best_val,
    test_results,
):

    benchmark_dir = os.path.join(
        RESULTS_DIR,
        benchmark,
    )

    os.makedirs(
        benchmark_dir,
        exist_ok=True,
    )

    torch.save(
        encoder.state_dict(),
        os.path.join(
            benchmark_dir,
            "encoder.pt",
        ),
    )

    torch.save(
        surrogate.state_dict(),
        os.path.join(
            benchmark_dir,
            "surrogate.pt",
        ),
    )

    results = {
        "benchmark": benchmark,
        "best_val": float(best_val),
        "train": [float(x) for x in train_history],
        "val": [float(x) for x in val_history],
        "test_mse": float(test_results["mse"]),
        "test_rmse": float(test_results["rmse"]),
        "test_mse_original": float(test_results["mse_original"]),
        "test_rmse_original": float(test_results["rmse_original"]),
    }

    with open(
        os.path.join(
            benchmark_dir,
            "results.json",
        ),
        "w",
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
        )

    return results


# ============================================================
# MAIN
# ============================================================


def main():

    print("\n========================================")
    print("QUANTUM TRANSFER EXPERIMENT")
    print("========================================")

    benchmarks = discover_benchmarks()

    for benchmark, paths in benchmarks.items():

        print("\n\n========================================")
        print("BENCHMARK:", benchmark)
        print("========================================")

        (
            train_loader,
            val_loader,
            test_loader,
            energy_mean,
            energy_std,
            input_mean,
            input_std,
        ) = load_data(
            paths["train"],
            paths["val"],
            paths["test"],
        )

        # ----------------------------------------------------
        # CRITICAL SANITY TEST
        # ----------------------------------------------------

        if RUN_OVERFIT_TEST:

            passed = run_overfit_test(train_loader)

            if not passed:

                raise RuntimeError(
                    "\n\nSTOPPING.\n"
                    "The encoder + surrogate cannot "
                    "memorize a tiny dataset.\n\n"
                    "Fix the model/data interface before "
                    "running the benchmark."
                )

        # ----------------------------------------------------
        # REAL TRAINING
        # ----------------------------------------------------

        (
            encoder,
            surrogate,
            train_history,
            val_history,
            best_val,
        ) = train_model(
            train_loader,
            val_loader,
            benchmark,
        )

        # ----------------------------------------------------
        # TEST
        # ----------------------------------------------------

        test_results = evaluate(
            encoder,
            surrogate,
            test_loader,
            energy_mean,
            energy_std,
        )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        results = save_results(
            benchmark,
            encoder,
            surrogate,
            train_history,
            val_history,
            best_val,
            test_results,
        )

        # ----------------------------------------------------
        # REPORT
        # ----------------------------------------------------

        print("\nRESULT:", benchmark)

        print(
            "Best validation MSE:",
            f"{best_val:.6f}",
        )

        print(
            "Test MSE:",
            f"{test_results['mse']:.6f}",
        )

        print(
            "Test RMSE:",
            f"{test_results['rmse']:.6f}",
        )

        print(
            "Test MSE original:",
            f"{test_results['mse_original']:.6f}",
        )

        print(
            "Test RMSE original:",
            f"{test_results['rmse_original']:.6f}",
        )

    print("\n========================================")
    print("DONE")
    print("========================================")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
