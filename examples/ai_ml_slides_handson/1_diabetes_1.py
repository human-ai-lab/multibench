# 1_diabetes_1.py

# 0. Load required packages/library
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import datasets, model_selection
from sklearn.metrics import mean_squared_error, r2_score

from _multibench_lesson_utils import (
    Concat,
    MLP,
    count_parameters,
    get_device,
    make_loader,
    predict_numpy,
    train_and_load,
)

SCRIPT_DIR = Path(__file__).resolve().parent
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# 1. Load dataset
# X contains 10 standardized input features; y is the disease progression score.
X_raw, y = datasets.load_diabetes(return_X_y=True)
print("Shape of Raw Input: ")
print(X_raw.shape)
print("First Sample: ")
print(X_raw[0])

# 2. Select the 3rd feature (BMI) so a simple 2D plot is possible.
X = X_raw[:, 2]
print("Shape of feature (old, 1D):")
print(X.shape)

# multibench/PyTorch models expect a 2D feature matrix: (n_samples, n_features).
X = X.reshape(-1, 1).astype(np.float32)
y = y.astype(np.float32)
print("Shape of feature (new, 2D):")
print(X.shape)

# 3. Split into train, validation, and test
# random_state makes the result reproducible for classroom demonstrations.
X_train_full, X_test, y_train_full, y_test = model_selection.train_test_split(
    X, y, test_size=0.33, random_state=SEED
)
X_train, X_valid, y_train, y_valid = model_selection.train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=SEED
)

train_loader = make_loader(
    X_train, y_train, task="regression", batch_size=32, shuffle=True, seed=SEED
)
valid_loader = make_loader(X_valid, y_valid, task="regression", batch_size=64)

# 4. Train a small multibench-style unimodal model
device = get_device()
encoders = [MLP(1, 16, 8, dropout=False).to(device)]
fusion = Concat().to(device)
head = MLP(8, 16, 1, dropout=False).to(device)
print(f"Using device: {device}")
print(f"Trainable parameters: {count_parameters([*encoders, fusion, head])}")

model, train_seconds = train_and_load(
    encoders,
    fusion,
    head,
    train_loader,
    valid_loader,
    epochs=100,
    task="regression",
    objective=torch.nn.MSELoss(),
    save_path=SCRIPT_DIR / ".multibench_models" / "diabetes_bmi.pt",
    lr=3e-3,
    weight_decay=1e-4,
)
print(f"Training time: {train_seconds:.2f}s")

# 5. Predict
y_pred = predict_numpy(model, X_test).reshape(-1)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)
print(f"Test RMSE: {rmse:.2f}")
print(f"Test R2: {r2:.3f}")

# 6. Plot
# Sort by the x-axis before drawing the line; otherwise matplotlib connects
# predictions in random test-set order, producing a misleading zig-zag line.
sort_index = np.argsort(X_test.ravel())
plt.figure(figsize=(7, 5))
plt.scatter(X_test, y_test, color="black", label="Actual test data")
plt.plot(
    X_test[sort_index],
    y_pred[sort_index],
    color="blue",
    linewidth=3,
    label="Multibench MLP prediction",
)
plt.xlabel("BMI (standardized)")
plt.ylabel("Diabetes progression")
plt.title("Multibench regression using one diabetes feature")
plt.legend()
output_path = SCRIPT_DIR / "1_diabetes_1_plot.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved plot to {output_path}")
