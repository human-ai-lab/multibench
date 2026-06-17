# 2_breast_cancer.py
# Reference: https://www.geeksforgeeks.org/machine-learning/ml-cancer-cell-classification-using-scikit-learn/

# 0. import library
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from _multibench_lesson_utils import (
    Concat,
    MLP,
    count_parameters,
    get_device,
    make_loader,
    predict_numpy,
    single_test,
    train_and_load,
)

SCRIPT_DIR = Path(__file__).resolve().parent
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

data = load_breast_cancer()

# 1. Exploring the dataset with Pandas
# Each row is one tumor sample; each column is a measured cell feature.
df = pd.DataFrame(data.data, columns=data.feature_names)

# Print the first 5 samples so students see the exact same rows each run.
print("First 5 samples of data:")
print(df.head(5))

print("Info of data:")
df.info()

print("Statistics of data:")
print(df.describe())

# Analyze data.target to understand the distribution of malignant and benign
# cases, since class imbalance can affect model performance.
target_names = {index: name for index, name in enumerate(data.target_names)}
df2 = pd.DataFrame(data.target, columns=["target"])
df2["diagnosis"] = df2["target"].map(target_names)

print("Class label mapping:")
print(target_names)

class_counts = df2["diagnosis"].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(
    class_counts,
    labels=class_counts.index,
    autopct="%1.2f%%",
    colors=["green", "red"],
)
plt.title("Breast cancer diagnosis distribution")
output_path = SCRIPT_DIR / "2_breast_cancer_class_distribution.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved class distribution plot to {output_path}")

# Split dataset into training, validation, and testing sets.
# stratify keeps the malignant/benign ratio similar in every split.
X_train_full, X_test, y_train_full, y_test = train_test_split(
    data.data, data.target, test_size=0.33, random_state=SEED, stratify=data.target
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.25,
    random_state=SEED,
    stratify=y_train_full,
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_valid = scaler.transform(X_valid).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

train_loader = make_loader(
    X_train, y_train, task="classification", batch_size=32, shuffle=True, seed=SEED
)
valid_loader = make_loader(X_valid, y_valid, task="classification", batch_size=64)
test_loader = make_loader(X_test, y_test, task="classification", batch_size=128)

# Train the multibench-style model. The dataset has one modality, so the encoder
# is unimodal and Concat simply forwards the encoded representation.
device = get_device()
encoders = [MLP(data.data.shape[1], 64, 32, dropout=True, dropoutp=0.1).to(device)]
fusion = Concat().to(device)
head = MLP(32, 32, len(data.target_names), dropout=False).to(device)
print(f"Using device: {device}")
print(f"Trainable parameters: {count_parameters([*encoders, fusion, head])}")

model, train_seconds = train_and_load(
    encoders,
    fusion,
    head,
    train_loader,
    valid_loader,
    epochs=30,
    task="classification",
    objective=torch.nn.CrossEntropyLoss(),
    save_path=SCRIPT_DIR / ".multibench_models" / "breast_cancer.pt",
    lr=1e-3,
    weight_decay=1e-4,
)
print(f"Training time: {train_seconds:.2f}s")

print("Testing:")
single_test(
    model,
    test_loader,
    criterion=torch.nn.CrossEntropyLoss(),
    task="classification",
)

y_logits = predict_numpy(model, X_test)
y_pred = np.argmax(y_logits, axis=1)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))
