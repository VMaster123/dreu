import os
import matplotlib.pyplot as plt

# --------------------------------------------------
# Validation MSE from ClassicalEncoder experiment
# --------------------------------------------------

encoder_val = [
    1.027455,
    1.026533,
    0.962511,
    0.954597,
    0.949648,
    0.945899,
    0.937008,
    0.933998,
    0.927245,
    0.921212,
    0.916069,
    0.915630,
    0.905818,
    0.886648,
    0.838457,
    0.773369,
    0.761330,
    0.745932,
    0.724199,
    0.712036,
    0.707765,
    0.715480,
    0.707592,
    0.703978,
    0.698166,
    0.701711,
    0.702697,
    0.704965,
    0.704486,
    0.710956,
    0.709531,
    0.713381,
    0.706033,
    0.712984,
    0.713900,
    0.704611,
    0.713858,
    0.713291,
    0.708518,
    0.708621,
    0.728610,
    0.712950,
    0.717306,
    0.719443,
    0.717326,
]


# --------------------------------------------------
# Validation MSE from no-encoder experiment
# --------------------------------------------------

no_encoder_val = [
    1.025881,
    0.932774,
    0.874205,
    0.829325,
    0.795812,
    0.765071,
    0.747295,
    0.747286,
    0.738802,
    0.734582,
    0.731548,
    0.730840,
    0.727690,
    0.726875,
    0.730195,
    0.724977,
    0.726267,
    0.724524,
    0.721214,
    0.725135,
    0.728490,
    0.729313,
    0.724348,
    0.725527,
    0.729114,
    0.723212,
    0.723391,
    0.725990,
    0.728996,
    0.727944,
    0.723260,
    0.735646,
    0.727522,
    0.743982,
    0.741253,
    0.734992,
    0.736660,
    0.739523,
    0.740714,
]


# --------------------------------------------------
# Epoch numbers
# --------------------------------------------------

encoder_epochs = range(1, len(encoder_val) + 1)
no_encoder_epochs = range(1, len(no_encoder_val) + 1)


# --------------------------------------------------
# Best validation values
# --------------------------------------------------

encoder_best = min(encoder_val)
encoder_best_epoch = encoder_val.index(encoder_best) + 1

no_encoder_best = min(no_encoder_val)
no_encoder_best_epoch = no_encoder_val.index(no_encoder_best) + 1


# --------------------------------------------------
# Plot
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    encoder_epochs,
    encoder_val,
    label="ClassicalEncoder + Surrogate",
    linewidth=2,
)

plt.plot(
    no_encoder_epochs,
    no_encoder_val,
    label="No Encoder: Raw 57 Features",
    linewidth=2,
)


# Mark best encoder point
plt.scatter(
    encoder_best_epoch,
    encoder_best,
    s=70,
    zorder=5,
)

plt.annotate(
    f"Best: {encoder_best:.4f}\nEpoch {encoder_best_epoch}",
    (encoder_best_epoch, encoder_best),
    xytext=(10, -35),
    textcoords="offset points",
)


# Mark best no-encoder point
plt.scatter(
    no_encoder_best_epoch,
    no_encoder_best,
    s=70,
    zorder=5,
)

plt.annotate(
    f"Best: {no_encoder_best:.4f}\nEpoch {no_encoder_best_epoch}",
    (no_encoder_best_epoch, no_encoder_best),
    xytext=(10, 15),
    textcoords="offset points",
)


# --------------------------------------------------
# Formatting
# --------------------------------------------------

plt.xlabel("Epoch")
plt.ylabel("Validation MSE")
plt.title("Validation Performance: Encoder vs. No Encoder")

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

# Save in the same folder as this Python script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_PATH = os.path.join(
    SCRIPT_DIR,
    "encoder_vs_no_encoder_validation.png",
)

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight",
)

print(f"\nGraph saved to:")
print(OUTPUT_PATH)

plt.show()
