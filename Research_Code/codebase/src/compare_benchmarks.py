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

# Number of additional epochs to train from the loaded
# checkpoint.
EPOCHS = 100

LR = 3e-4
PATIENCE = 5

LATENT_DIM = 64
WEIGHT_DECAY = 1e-4


# ============================================================
# CHECKPOINT SETTINGS
# ============================================================

# IMPORTANT:
#
# True  = LOAD EXISTING WEIGHTS. DO NOT START FROM SCRATCH.
# False = intentionally create a new model.
#
# Keep this TRUE.
RESUME_FROM_CHECKPOINT = True


# If a checkpoint is missing while RESUME_FROM_CHECKPOINT=True,
# the program STOPS rather than silently training from scratch.
REQUIRE_CHECKPOINT = True


# ============================================================
# DEBUG
# ============================================================

DEBUG_MODE = False

DEBUG_TRAIN_SAMPLES = 2048
DEBUG_VAL_SAMPLES = 2048


# ============================================================
# OVERFIT TEST
# ============================================================

# DO NOT run the fresh-model overfit test when resuming.
#
# The old version created a brand-new random encoder and
# surrogate here, which has nothing to do with your existing
# trained model.
#
# If you specifically want to perform a fresh architecture
# sanity test later, turn this on manually.
RUN_OVERFIT_TEST = False

OVERFIT_SAMPLES = 128
OVERFIT_EPOCHS = 1000
OVERFIT_LR = 3e-3


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
    Surrogate model.

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
# CHECKPOINT PATHS
# ============================================================


def get_checkpoint_paths(benchmark):

    benchmark_dir = os.path.join(
        RESULTS_DIR,
        benchmark,
    )

    encoder_path = os.path.join(
        benchmark_dir,
        "encoder.pt",
    )

    surrogate_path = os.path.join(
        benchmark_dir,
        "surrogate.pt",
    )

    return (
        benchmark_dir,
        encoder_path,
        surrogate_path,
    )


# ============================================================
# MODEL CREATION
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
# LOAD EXISTING CHECKPOINT
# ============================================================


def load_existing_models(benchmark):

    (
        benchmark_dir,
        encoder_path,
        surrogate_path,
    ) = get_checkpoint_paths(benchmark)

    print("\n" + "=" * 60)
    print("CHECKPOINT STATUS")
    print("=" * 60)

    print(
        "Benchmark:",
        benchmark,
    )

    print(
        "Encoder checkpoint:",
        encoder_path,
    )

    print(
        "Surrogate checkpoint:",
        surrogate_path,
    )

    encoder_exists = os.path.exists(encoder_path)

    surrogate_exists = os.path.exists(surrogate_path)

    print(
        "Encoder exists:",
        encoder_exists,
    )

    print(
        "Surrogate exists:",
        surrogate_exists,
    )

    # --------------------------------------------------------
    # PROTECTION AGAINST ACCIDENTALLY STARTING FROM SCRATCH
    # --------------------------------------------------------

    if REQUIRE_CHECKPOINT:

        if not encoder_exists:
            raise FileNotFoundError(
                "\nSTOPPING: Existing encoder checkpoint "
                "was not found.\n\n"
                f"Expected:\n{encoder_path}\n\n"
                "The script will NOT create a fresh encoder "
                "because REQUIRE_CHECKPOINT=True."
            )

        if not surrogate_exists:
            raise FileNotFoundError(
                "\nSTOPPING: Existing surrogate checkpoint "
                "was not found.\n\n"
                f"Expected:\n{surrogate_path}\n\n"
                "The script will NOT create a fresh surrogate "
                "because REQUIRE_CHECKPOINT=True."
            )

    encoder = create_encoder()
    surrogate = create_surrogate()

    # --------------------------------------------------------
    # LOAD WEIGHTS
    # --------------------------------------------------------

    print("\nLoading existing encoder weights...")

    encoder_state = torch.load(
        encoder_path,
        map_location=DEVICE,
    )

    encoder.load_state_dict(encoder_state)

    print("Encoder weights loaded.")

    print("\nLoading existing surrogate weights...")

    surrogate_state = torch.load(
        surrogate_path,
        map_location=DEVICE,
    )

    surrogate.load_state_dict(surrogate_state)

    print("Surrogate weights loaded.")

    print(
        "\n*** MODEL IS BEING RESUMED FROM EXISTING "
        "WEIGHTS — NOT FROM RANDOM INITIALIZATION. ***"
    )

    return encoder, surrogate


# ============================================================
# CREATE NEW MODELS
# ============================================================


def create_new_models():

    if REQUIRE_CHECKPOINT:
        raise RuntimeError(
            "Refusing to create a new model because " "REQUIRE_CHECKPOINT=True."
        )

    print("\nWARNING: Creating NEW models from random " "initialization.")

    encoder = create_encoder()
    surrogate = create_surrogate()

    return encoder, surrogate


# ============================================================
# MODEL INITIALIZATION
# ============================================================


def initialize_models(benchmark):

    if RESUME_FROM_CHECKPOINT:

        return load_existing_models(benchmark)

    return create_new_models()


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
# OPTIONAL OVERFIT TEST
# ============================================================


def run_overfit_test(train_loader):

    print("\n" + "=" * 60)
    print("OVERFIT SANITY TEST")
    print("=" * 60)

    print("WARNING: This creates a COMPLETELY NEW model.")

    batches = []

    total = 0

    for batch in train_loader:

        batch_size = batch["energy"].shape[0]

        remaining = OVERFIT_SAMPLES - total

        if remaining <= 0:
            break

        if batch_size > remaining:
            batch = {k: v[:remaining] for k, v in batch.items()}

        batches.append(batch)

        total += batch["energy"].shape[0]

        if total >= OVERFIT_SAMPLES:
            break

    if total < OVERFIT_SAMPLES:
        raise RuntimeError(f"Could only collect {total} samples.")

    fixed_batch = {}

    for key in batches[0].keys():

        values = [b[key] for b in batches]

        try:
            fixed_batch[key] = torch.cat(
                values,
                dim=0,
            )
        except RuntimeError:
            fixed_batch[key] = values[0]

    fixed_batch = {k: v.to(DEVICE) for k, v in fixed_batch.items()}

    encoder = create_encoder()
    surrogate = create_surrogate()

    encoder.train()
    surrogate.train()

    parameters = list(encoder.parameters()) + list(surrogate.parameters())

    optimizer = torch.optim.Adam(
        parameters,
        lr=OVERFIT_LR,
        weight_decay=0.0,
    )

    criterion = nn.MSELoss()

    final_loss = float("inf")

    for epoch in range(OVERFIT_EPOCHS):

        optimizer.zero_grad(set_to_none=True)

        prediction, _ = forward_encoder(
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

        if epoch == 0 or (epoch + 1) % 50 == 0:

            print(
                f"Overfit Epoch "
                f"{epoch + 1:04d}/"
                f"{OVERFIT_EPOCHS} | "
                f"MSE = {final_loss:.8f}"
            )

        if final_loss < 2e-3:

            print("\nOVERFIT TEST PASSED.")

            return True

    print("\nOVERFIT TEST FAILED.")

    print(f"Final MSE: {final_loss:.8f}")

    return False


# ============================================================
# TRAIN
# ============================================================


def train_model(
    encoder,
    surrogate,
    train_loader,
    val_loader,
    benchmark,
):

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

    if RESUME_FROM_CHECKPOINT:

        print("Starting from EXISTING checkpoint.")

    else:

        print("Starting from RANDOM initialization.")

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

        raise RuntimeError("Training never produced a " "valid checkpoint.")

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

    (
        benchmark_dir,
        encoder_path,
        surrogate_path,
    ) = get_checkpoint_paths(benchmark)

    os.makedirs(
        benchmark_dir,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # SAVE ENCODER
    # --------------------------------------------------------

    torch.save(
        encoder.state_dict(),
        encoder_path,
    )

    # --------------------------------------------------------
    # SAVE SURROGATE
    # --------------------------------------------------------

    torch.save(
        surrogate.state_dict(),
        surrogate_path,
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

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

    results_path = os.path.join(
        benchmark_dir,
        "results.json",
    )

    with open(
        results_path,
        "w",
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
        )

    print(
        "\nSaved encoder:",
        encoder_path,
    )

    print(
        "Saved surrogate:",
        surrogate_path,
    )

    print(
        "Saved results:",
        results_path,
    )

    return results


# ============================================================
# MAIN
# ============================================================


def main():

    print("\n========================================")
    print("QUANTUM TRANSFER EXPERIMENT")
    print("========================================")

    print(
        "\nRESUME_FROM_CHECKPOINT =",
        RESUME_FROM_CHECKPOINT,
    )

    print(
        "REQUIRE_CHECKPOINT =",
        REQUIRE_CHECKPOINT,
    )

    if RESUME_FROM_CHECKPOINT and REQUIRE_CHECKPOINT:

        print("\n*** SAFETY MODE ENABLED ***")

        print("Existing weights are required.")

        print("The script will NEVER silently " "start from scratch.")

    benchmarks = discover_benchmarks()

    for benchmark, paths in benchmarks.items():

        print("\n\n========================================")

        print(
            "BENCHMARK:",
            benchmark,
        )

        print("========================================")

        # ----------------------------------------------------
        # DATA
        # ----------------------------------------------------

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
        # LOAD EXISTING MODEL
        # ----------------------------------------------------

        encoder, surrogate = initialize_models(benchmark)

        # ----------------------------------------------------
        # OPTIONAL OVERFIT TEST
        # ----------------------------------------------------

        if RUN_OVERFIT_TEST:

            passed = run_overfit_test(train_loader)

            if not passed:

                raise RuntimeError(
                    "\n\nSTOPPING.\n"
                    "The fresh encoder + surrogate "
                    "cannot memorize a tiny dataset.\n\n"
                    "Fix the model/data interface "
                    "before continuing."
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
            encoder,
            surrogate,
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

        print(
            "\nRESULT:",
            benchmark,
        )

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
