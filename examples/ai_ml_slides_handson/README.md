# AI/ML Slides Hands-on for Multibench

This directory is the **Multibench version** of the hands-on examples from
[bagustris/ai-ml-slides](https://github.com/bagustris/ai-ml-slides).

The original materials introduce AI and machine learning examples for teaching.
This version adapts the examples to run with a local
[Multibench](https://github.com/pliang279/MultiBench) checkout by training
small PyTorch models through Multibench encoders, fusion modules, and supervised
learning utilities.

## Contents

- `1_diabetes_1.py` - regression on the scikit-learn diabetes dataset using one
  BMI feature and a small Multibench-style MLP.
- `2_breast_cancer.py` - binary classification on the scikit-learn breast cancer
  dataset with a Multibench-style unimodal model.
- `3_diabetes_shap_1.py` - diabetes regression with SHAP explanations for the
  trained model.
- `_multibench_lesson_utils.py` - local helper functions that connect the lesson
  scripts to a Multibench checkout.
- `requirements.txt` - Python packages needed by the lesson scripts.

## Setup

Run these examples from inside a local Multibench repository, or set
`MULTIBENCH_ROOT` to the path of your Multibench checkout.

```bash
cd /path/to/multibench/examples/ai_ml_slides_handson
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If the scripts cannot find Multibench automatically, set:

```bash
export MULTIBENCH_ROOT=/path/to/multibench
```

## Run

```bash
python 1_diabetes_1.py
python 2_breast_cancer.py
python 3_diabetes_shap_1.py
```

The scripts train small models, print metrics, and save generated outputs in
this directory. Model checkpoints are written under `.multibench_models/`.

Expected generated files include:

- `1_diabetes_1_plot.png`
- `2_breast_cancer_class_distribution.png`
- `3_diabetes_shap_1_summary.png`
- `3_diabetes_shap_1_bmi_dependence.png`
- `3_diabetes_shap_1_force_single.html`
- `3_diabetes_shap_1_force_all.html`

## Notes

These scripts are intended as teaching examples. They favor readable,
step-by-step code over maximum model performance, while preserving the main
Multibench workflow: dataset preparation, unimodal encoding, fusion, supervised
training, prediction, evaluation, and explanation.
