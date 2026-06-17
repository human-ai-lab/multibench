import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

try:
    import shap
except ModuleNotFoundError as exc:
    raise SystemExit(
        "This lesson needs shap. Install the lesson requirements, then rerun this script."
    ) from exc

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


def finish_plot(filename):
    """Save plots in headless mode, otherwise show them interactively."""
    if plt.get_backend().lower() == "agg":
        output_path = SCRIPT_DIR / filename
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
        plt.close()
    else:
        plt.show()


def save_force_plot(filename, force_plot):
    """Write SHAP force plots to HTML files for script-based runs."""
    output_path = SCRIPT_DIR / filename
    with output_path.open("w", encoding="utf-8") as html_file:
        shap.save_html(html_file, force_plot)
    print(f"Saved force plot to {output_path}")


class ShapReadyModel(torch.nn.Module):
    """Adapter so SHAP can call a multibench MMDL model with one tensor."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        encoded = self.model.encoders[0](x)
        return self.model.head(encoded)


def normalize_shap_values(values):
    """Return SHAP values as an (n_samples, n_features) numpy array."""
    if isinstance(values, list):
        values = values[0]
    values = np.asarray(values)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    return values


# X is a pandas DataFrame of diabetes features; y is the target progression score.
X, y = shap.datasets.diabetes()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=SEED
)

X_train_np = X_train.to_numpy(dtype=np.float32)
X_valid_np = X_valid.to_numpy(dtype=np.float32)
X_test_np = X_test.to_numpy(dtype=np.float32)
y_train_np = np.asarray(y_train, dtype=np.float32)
y_valid_np = np.asarray(y_valid, dtype=np.float32)

train_loader = make_loader(
    X_train_np, y_train_np, task="regression", batch_size=32, shuffle=True, seed=SEED
)
valid_loader = make_loader(X_valid_np, y_valid_np, task="regression", batch_size=64)

device = get_device()
encoders = [MLP(X_train_np.shape[1], 64, 32, dropout=True, dropoutp=0.05).to(device)]
fusion = Concat().to(device)
head = MLP(32, 32, 1, dropout=False).to(device)
print(f"Using device: {device}")
print(f"Trainable parameters: {count_parameters([*encoders, fusion, head])}")

model, train_seconds = train_and_load(
    encoders,
    fusion,
    head,
    train_loader,
    valid_loader,
    epochs=80,
    task="regression",
    objective=torch.nn.MSELoss(),
    save_path=SCRIPT_DIR / ".multibench_models" / "diabetes_shap.pt",
    lr=1e-3,
    weight_decay=1e-4,
)
print(f"Training time: {train_seconds:.2f}s")


def print_error(predict_fn):
    """Print RMSE so students can connect model accuracy with explanations."""
    y_pred = predict_fn(X_test_np).reshape(-1)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    print(f"Root mean squared test error = {rmse:.2f}")
    time.sleep(0.5)  # to let the print get out before any progress bars


print_error(lambda values: predict_numpy(model, values))

# DeepExplainer uses the PyTorch graph directly and avoids Kernel SHAP's many
# repeated model calls. A small background sample is enough for this lesson.
model = model.to("cpu").eval()
shap_model = ShapReadyModel(model).eval()
background = torch.as_tensor(X_train_np[:64], dtype=torch.float32)
to_explain = torch.as_tensor(X_test_np, dtype=torch.float32)
explainer = shap.DeepExplainer(shap_model, background)

try:
    raw_shap_values = explainer.shap_values(to_explain, check_additivity=False)
except TypeError:
    raw_shap_values = explainer.shap_values(to_explain)

shap_values = normalize_shap_values(raw_shap_values)
expected_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])

# Explain one prediction first; this shows how each feature moves the prediction
# away from the model's average prediction.
single_force_plot = shap.force_plot(expected_value, shap_values[0], X_test.iloc[0, :])
save_force_plot("3_diabetes_shap_1_force_single.html", single_force_plot)

# Explain the model's predictions on the whole test set.
shap.summary_plot(shap_values, X_test, show=False)
finish_plot("3_diabetes_shap_1_summary.png")

# plot the SHAP values for a single feature (bmi)
shap.dependence_plot("bmi", shap_values, X_test, show=False)
finish_plot("3_diabetes_shap_1_bmi_dependence.png")

# Force plot for the whole test set. In a notebook this renders inline; when
# running as a script, save the returned HTML object if you need a shareable file.
force_plot = shap.force_plot(expected_value, shap_values, X_test)
save_force_plot("3_diabetes_shap_1_force_all.html", force_plot)

# Challenge: try using a different models and see how the explanations differ!
# For example, try a deeper head, a smaller encoder, or a different fusion layer.
