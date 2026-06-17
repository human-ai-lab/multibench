# MultiBench

**Standardized toolkit for multimodal deep learning research**

[Documentation](https://human-ai-lab.github.io/multibench/)   
![Overview](images/overview.png)

MultiBench is a systematic, unified large-scale benchmark spanning **15 datasets**, **10 modalities**, **20 prediction tasks**, and **6 research areas**. It provides an automated end-to-end ML pipeline that simplifies data loading, experimental setup, and model evaluation.

Paired with **MultiZoo**, a modular collection of 20 core multimodal learning approaches, MultiBench holistically evaluates:

1. **Performance** across domains and modalities
2. **Complexity** during training and inference
3. **Robustness** to noisy and missing modalities

## Supported datasets

| Research Area        | Datasets                                           |
| -------------------- | -------------------------------------------------- |
| Affective Computing  | CMU-MOSI, CMU-MOSEI, MUStARD, UR-FUNNY            |
| Healthcare           | MIMIC                                              |
| Robotics             | Vision & Touch, MuJoCo Push (Gentle Push)          |
| Finance              | Stocks-Food, Stocks-Health, Stocks-Tech            |
| HCI                  | ENRICO                                             |
| Multimedia           | AV-MNIST, MM-IMDb, Kinetics-S, Kinetics-L         |

![Datasets](images/datasets.png)

## Supported algorithms

![MultiZoo](images/multizoo.png)

- **Unimodal models** — MLP, GRU, LSTM, CNN, LeNet, ResNet, Transformer, FCN, Random Forest, etc. (`unimodals/`)
- **Fusion paradigms** — Early/Late Fusion, Tensor Fusion, Multiplicative Interactions, Low-Rank Tensor Fusion, NL-Gate, MulT, etc. (`fusions/`)
- **Objective functions** — CrossEntropy, MSE, ELBO, CCA, Contrastive Loss, Reconstruction Loss, etc. (`objective_functions/`)
- **Training structures** — Supervised Learning (Early/Late Fusion, MVAE, MFM), Gradient Blend, Architecture Search, MCTN, etc. (`training_structures/`)

## Getting started

### Prerequisites

- Python >= 3.10
- PyTorch (CPU or GPU)

### Virtual environment  
It is advised to use a virtual environment to manage dependencies. You can create one using `uv venv` (recommended) or `venv`:

```bash
uv venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```
If not using uv, you can create a virtual environment using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```
And remove `uv` from the installation commands below.

### Installation
For GPU support, install using the appropriate command from the [PyTorch website](https://pytorch.org/get-started/locally/). The following command installs the latest stable version of PyTorch with CUDA latest support:

```bash
uv pip install -r requirements.txt
```

For CPU, install directly:

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install memory-profiler scikit-learn scipy matplotlib h5py tqdm
```

The finance examples below also use the optional `finance` dependency group:

```bash
uv sync --extra finance
```

Other dataset-specific examples may require packages such as `gdown`,
`fannypack`, or `numpy<2`.

### Quick example

This example trains a simple early fusion model on the [CMU-MOSI](https://drive.google.com/drive/folders/1uEK737LXB9jAlf9kyqRs6B9N6cDncodq?usp=sharing) sentiment dataset.

```bash
# Download the MOSI dataset
pip install gdown
gdown https://drive.google.com/u/0/uc?id=1szKIqO0t3Be_W91xvf6aYmsVVUa7wDHU
mkdir -p data/affect && mv mosi_raw.pkl data/affect/
```

```python
import torch
from datasets.affect.get_data import get_dataloader
from unimodals.common_models import GRU, MLP, Sequential, Identity
from fusions.common_fusions import ConcatEarly
from training_structures.Supervised_Learning import train, test
from utils.device import get_device

device = get_device()  # automatically selects CUDA, MPS (Apple Silicon), or CPU

# Load data (3 modalities: text, audio, vision)
traindata, validdata, testdata = get_dataloader(
    'data/affect/mosi_raw.pkl',
    data_type='mosi', max_pad=True, max_seq_len=50
)

# Define model components
encoders = [Identity().to(device) for _ in range(3)]
fusion = ConcatEarly().to(device)
head = Sequential(
    GRU(409, 512, dropout=True, has_padding=False, batch_first=True, last_only=True),
    MLP(512, 512, 1)
).to(device)

# Train
train(encoders, fusion, head, traindata, validdata,
      total_epochs=10, task="regression",
      optimtype=torch.optim.AdamW, lr=1e-3,
      save='results/models/mosi_ef_r0.pt', objective=torch.nn.L1Loss())

# Test
model = torch.load('results/models/mosi_ef_r0.pt', weights_only=False).to(device)
test(model, testdata, dataset='affect', is_packed=False,
     criterion=torch.nn.L1Loss(), task="posneg-classification", no_robust=True)
```

> [!TIP]
> Check out the [Colab tutorials](https://colab.research.google.com/drive/1PfgotUdIWQvA0mIUbncLPSRzqAPXKZV9) for step-by-step walkthroughs covering basic early fusion. You then can continue to architecture search (MFAS) and cross-modal translation (MCTN).

## Running experiments

Each dataset has dedicated example scripts under `examples/`. Here are some common ones:

```bash
# Affective computing
python examples/affect/affect_late_fusion.py

# Healthcare (requires MIMIC access — see dataset section below)
python examples/healthcare/mimic_low_rank_tensor.py

# Robotics
python examples/robotics/LRTF.py
python examples/gentle_push/LF.py

# Finance (specify input and target stocks)
uv run --extra finance python examples/finance/stocks_late_fusion.py --input-stocks 'AAPL MSFT AMZN INTC AMD MSI' --target-stock 'MSFT'

# HCI
python examples/hci/enrico_simple_late_fusion.py

# Multimedia
python examples/multimedia/avmnist_simple_late_fusion.py
python examples/multimedia/mmimdb_simple_late_fusion.py
```

### Quickest experiments to get started

If you just want to confirm your install works and see the full
data → train → evaluate pipeline run end to end, these are the fastest
entry points. All run on **CPU** with real data and the default 2-epoch
example settings, except the MOSI code block above, which uses 10 epochs.

| Experiment | Script | Data | Approx. CPU runtime | Model params |
| ---------- | ------ | ---- | ------------------- | ------------ |
| Stock prediction | `examples/finance/stocks_late_fusion.py` | Auto-downloads via `yfinance` | ~20 s | 7.4 K |
| AV-MNIST (late fusion) | `examples/multimedia/avmnist_simple_late_fusion.py` | 2,000 real training examples | ~26 s | 260.9 K |
| Gentle Push (unimodal) | `examples/gentle_push/unimodal_image.py --quick` | 10 real train / val / test trajectories | ~36 s | 3.9 M |

**Smallest / fastest overall:** Stock prediction needs no manual download
(data is fetched on first run via `yfinance`) and finishes in seconds,
making it the best choice for a first smoke test or for quickly iterating
on model architecture. AV-MNIST is the simplest *multimodal* starting point —
its example already subsets to 2,000 training samples and 2 epochs
(`examples/multimedia/avmnist_simple_late_fusion.py`).

The Gentle Push script without `--quick` trains on the full
`gentle_push_1000.hdf5` training file and is CPU-compatible, but it is not a
quick smoke test on typical CPU-only machines.

Measured real-data CPU smoke-test results on this PC:

| Metric | Stock | AV-MNIST | Gentle Push `--quick` |
| ------ | ----- | -------- | --------------------- |
| Total runtime | 19.8 s | 25.8 s | 35.8 s |
| Training time | 8.4 s | 12.1 s | 27.5 s |
| Inference time | 0.27 s | 6.15 s | 2.84 s |
| Model parameters | 7,393 | 260,922 | 3,879,898 |
| Smoke-test metric | MSE 1.2406 | Accuracy 0.5499 | MSE 0.3309 |

These are quick pipeline checks, not benchmark-quality accuracy numbers.
Random initialization, data-fetch latency, and CPU model can move the results.

## Dataset access

Most datasets require a one-time download. Links to processed data:

| Dataset | Download |
| ------- | -------- |
| CMU-MOSI | [Google Drive](https://drive.google.com/drive/folders/1uEK737LXB9jAlf9kyqRs6B9N6cDncodq?usp=sharing) |
| CMU-MOSEI | [Google Drive](https://drive.google.com/drive/folders/1A_hTmifi824gypelGobgl2M-5Rw9VWHv?usp=sharing) |
| MUStARD | [Google Drive](https://drive.google.com/drive/folders/1JFcX-NF97zu9ZOZGALGU9kp8dwkP7aJ7?usp=sharing) |
| UR-FUNNY | [Google Drive](https://drive.google.com/drive/folders/1Agzm157lciMONHOHemHRSySmjn1ahHX1?usp=sharing) |
| AV-MNIST | [Google Drive](https://drive.google.com/file/d/1KvKynJJca5tDtI5Mmp6CoRh9pQywH8Xp/view?usp=sharing) |
| MM-IMDb | [Archive.org (hdf5)](https://archive.org/download/mmimdb/multimodal_imdb.hdf5) · [Archive.org (raw)](https://archive.org/download/mmimdb/mmimdb.tar.gz) |
| Vision & Touch | Run `datasets/robotics/download_data.sh` |
| MuJoCo Push | Auto-downloads on first run |
| Finance | Auto-downloads on first run |
| ENRICO | [GitHub](https://github.com/luileito/enrico) |
| Clotho | [GitHub](https://github.com/audio-captioning/clotho-dataset) |

> [!NOTE]
> **MIMIC** has restricted access. Follow [these instructions](https://mimic.mit.edu/iv/access/) to obtain credentials, then email yiweilyu@umich.edu with proof to request the preprocessed `im.pk` file.

## Evaluation

### Complexity

Track peak memory, parameter count, and runtime using `eval_scripts/complexity.py`:

```python
from eval_scripts.complexity import all_in_one_train, all_in_one_test
```

See `examples/healthcare/mimic_baseline_track_complexity.py` for a full example.

### Robustness

Modality-specific noise implementations are in `robustness/`, with evaluation scripts in `eval_scripts/robustness.py`. Robustness testing is integrated into the standard training/testing pipeline.

![Robustness plots](images/robustness_plots.png)

## Utilities

### Device abstraction

`utils/device.py` provides `get_device()`, which automatically selects the best available hardware — CUDA GPU, Apple Silicon MPS, or CPU — without any manual configuration. All core modules use this internally, so the toolkit runs on any hardware out of the box.

```python
from utils.device import get_device

device = get_device()          # "cuda:0", "mps", or "cpu"
model = MyModel().to(device)
```

### Shape validation

`utils/verify.py` provides `validate_shapes()`, which dry-runs the full encoder → fusion → head pipeline with sample tensors to catch dimension mismatches before training starts.

```python
from utils.verify import validate_shapes
import torch

sample_inputs = [torch.zeros(2, 1, 28, 28), torch.zeros(2, 1, 112, 112)]
validate_shapes(encoders, fusion, head, sample_inputs)
# Prints shapes at each stage; raises RuntimeError on mismatch
```

## Project structure

```
multibench/
├── datasets/              # Data loaders for all 15 datasets
├── unimodals/             # Single-modality encoders
├── fusions/               # Multimodal fusion strategies
├── objective_functions/   # Loss functions and objectives
├── training_structures/   # Training paradigms (supervised, NAS, etc.)
├── eval_scripts/          # Performance, complexity, robustness evaluation
├── robustness/            # Modality-specific noise implementations
├── examples/              # Runnable scripts and Colab notebooks
├── pretrained/            # Pre-trained model weights
├── results/               # Experiment outputs (gitignored, kept tidy here)
│   ├── models/            # Trained model checkpoints (*.pt) saved by examples
│   └── images/            # Robustness / accuracy plots (*.png)
└── utils/
    ├── device.py          # get_device(): selects CUDA → MPS → CPU automatically
    └── verify.py          # validate_shapes(): dry-runs pipeline to catch dim mismatches
```

> [!NOTE]
> Example scripts and the training structures save checkpoints to
> `results/models/` and robustness plots to `results/images/` by default.
> These output files are gitignored (only the folders are kept via
> `.gitkeep`), so your experiment artifacts stay in one place instead of
> scattering across the repo root.

## Adding new datasets or algorithms

**New dataset:**

1. Add a folder under `datasets/` with a `get_data.py` containing a `get_dataloader` function that returns `(train, valid, test)` dataloaders
2. Add example scripts under `examples/`

**New algorithm:**

1. Add your module to the appropriate subfolder: `unimodals/`, `fusions/`, `objective_functions/`, or `training_structures/`
2. Add example scripts and document input/output formats

## Citation

If you use MultiBench in your research, please cite:

```bibtex
@article{liang2023multizoo,
  title={MULTIZOO \& MULTIBENCH: A Standardized Toolkit for Multimodal Deep Learning},
  author={Liang, Paul Pu and Lyu, Yiwei and Fan, Xiang and Agarwal, Arav and Cheng, Yun and Morency, Louis-Philippe and Salakhutdinov, Ruslan},
  journal={Journal of Machine Learning Research},
  volume={24},
  pages={1--7},
  year={2023}
}
```

```bibtex
@inproceedings{liang2021multibench,
  title={MultiBench: Multiscale Benchmarks for Multimodal Representation Learning},
  author={Liang, Paul Pu and Lyu, Yiwei and Fan, Xiang and Wu, Zetian and Cheng, Yun and Wu, Jason and Chen, Leslie Yufan and Wu, Peter and Lee, Michelle A and Zhu, Yuke and others},
  booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1)},
  year={2021}
}
```

## Course adoption
MultiBench is used in the curriculum of the following courses: 
- [Human AI Interaction](https://bagustris.github.io/multisensory)

## Contributors

[Paul Pu Liang](http://www.cs.cmu.edu/~pliang/) · [Yiwei Lyu](https://github.com/lvyiwei1) · [Xiang Fan](https://github.com/sfanxiang) · [Zetian Wu](http://neal-ztwu.github.io) · [Yun Cheng](https://kapikantzari.github.io) · [Arav Agarwal](https://www.linkedin.com/in/arav-agarwal-941b44109/) · [Jason Wu](https://jasonwunix.com/) · Leslie Chen · [Peter Wu](https://peter.onrender.com/) · [Michelle A. Lee](http://stanford.edu/~mishlee/) · [Yuke Zhu](https://www.cs.utexas.edu/~yukez/) · [Ruslan Salakhutdinov](https://www.cs.cmu.edu/~rsalakhu/) · [Louis-Philippe Morency](https://www.cs.cmu.edu/~morency/)
