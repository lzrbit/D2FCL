# DynDFCL

**Dynamic Decentralized Federated Continual Learning**

Reference implementation for the paper *Decentralized Federated Continual
Learning with Dynamic Collaboration*. DynDFCL augments the DCFCL backbone
with three modules:

1. **Edge-side replay (DER++)** &mdash; a persistent per-client buffer that
   rehearses stored labels and output logits to suppress catastrophic
   forgetting.
2. **Directed coalition aggregation** &mdash; per-round asymmetric, task-aware
   mixing weights derived from gradient signatures.
3. **Dynamic directed mask** &mdash; a round-by-round trust mask that prunes
   peers whose updates would harm the receiving client.

The three modules can be ablated independently, and ablation scripts for
each are included in `scripts/`.

## Repository layout

```
DynDFCL/
├── main.py                  # Single-run entry point (CLI / YAML config)
├── run_experiments.py       # Multi-run launcher
├── configs/                 # Default YAML configurations
│   ├── dyndfcl_emnist.yaml
│   ├── dyndfcl_cifar100.yaml
│   ├── dcfcl_emnist.yaml
│   ├── dcfcl_cifar100.yaml
│   └── dcfcl_directed_emnist.yaml
├── core/                    # Framework
│   ├── config.py            # Config dataclass + CLI parsing
│   ├── server.py            # Decentralized server / coalition logic
│   ├── client.py            # Backward-compat shim (re-exports FL_model)
│   ├── models.py            # SimpleCNN / ResNet-18 + CBAM backbones
│   ├── optimizers.py        # SCAFFOLD / PerAvg / pFedMe optimizers
│   └── replay_buffer.py     # DER++ replay buffer
├── FL_model/                # Algorithm-specific client implementations
│   ├── base_client.py       # Shared client infrastructure
│   ├── fedavg.py            # FedAvg / Local
│   ├── fedprox.py           # FedProx
│   ├── fedlwf.py            # FedLwF (LwF + federated)
│   ├── scaffold.py          # SCAFFOLD
│   ├── peravg.py            # Per-FedAvg (MAML-based)
│   ├── pfedme.py            # pFedMe (Moreau-envelope personalization)
│   ├── dcfcl.py             # DCFCL baseline
│   └── dyndfcl.py           # DynDFCL (this paper)
├── utils/                   # Data loading + helpers
└── scripts/                 # Paper-reproduction scripts
    ├── run_full_training.py
    ├── run_all_algorithms.py
    ├── run_module_ablation.py        # Modules A / B / C ablation
    ├── run_coalition_ablation.py     # Coalition formation ablation
    ├── run_coalition_mask_test.py    # Coalition mask ablation
    ├── run_directed_comparison.py    # Symmetric vs directed
    ├── run_der_strength_ablation.py  # DER buffer-size sweep
    ├── run_emergence_evaluation.py   # Collective intelligence metrics
    ├── run_lambda_kd_study.py        # KD weight sweep
    ├── run_ablation.py               # General ablation harness
    ├── compute_round_timings.py      # Wall-clock measurement
    ├── generate_emnist_shuffle_split.py
    └── generate_cifar100_split.py
```

## Installation

```bash
# Option A: conda
conda env create -f environment.yml
conda activate fcl

# Option B: pip
pip install -r requirements.txt
```

Tested with Python 3.10, PyTorch 2.0+, on a single NVIDIA GPU.

## Quick start

### Reproduce the main DynDFCL result on EMNIST-Letters

```bash
python main.py --config configs/dyndfcl_emnist.yaml
```

This runs the full method (DER++ + directed coalition + coalition mask)
with the hyperparameters reported in the paper.

### Reproduce on CIFAR-100

```bash
python main.py --config configs/dyndfcl_cifar100.yaml
```

### Run a baseline

```bash
python main.py --algorithm FedAvg     --config configs/dcfcl_emnist.yaml
python main.py --algorithm FedProx    --config configs/dcfcl_emnist.yaml
python main.py --algorithm SCAFFOLD   --config configs/dcfcl_emnist.yaml
python main.py --algorithm pFedMe     --config configs/dcfcl_emnist.yaml
python main.py --algorithm DCFCL      --config configs/dcfcl_emnist.yaml
```

Supported algorithms (`--algorithm`):
`Local`, `FedAvg`, `FedProx`, `SCAFFOLD`, `FedLwF`, `PerAvg`, `pFedMe`,
`ClusterFL`, `DCFCL`, `DynDFCL`.

### Run all algorithms in one command

```bash
python scripts/run_all_algorithms.py
```

### Reproduce ablations

```bash
# Module ablation (DER / Directed / Mask)
python scripts/run_module_ablation.py                  # EMNIST
python scripts/run_module_ablation.py --dataset cifar100

# Coalition aggregation ablation (full / fedavg / random / singleton / global)
python scripts/run_coalition_ablation.py
python scripts/run_coalition_ablation.py --dataset cifar100

# DER++ buffer-size sweep
python scripts/run_der_strength_ablation.py

# Coalition mask ablation
python scripts/run_coalition_mask_test.py

# Directed vs symmetric collaboration
python scripts/run_directed_comparison.py

# Collective-intelligence emergence evaluation
python scripts/run_emergence_evaluation.py
```

## Configuration

YAML configs in `configs/` control all hyperparameters. CLI flags
overwrite YAML values, e.g.:

```bash
python main.py --config configs/dyndfcl_emnist.yaml \
               --lr 5e-5 --buffer_size 200 \
               --no_use_directed_collaboration
```

The most relevant flags are documented inline in
`configs/dyndfcl_emnist.yaml`.

## Datasets

`main.py` expects pre-computed split files under `split_files/` for each
dataset. The exact split files used in the paper are:

- EMNIST-Letters: `EMNIST_letters_split_cn8_tn6_cet2_cs2_s2571.pkl`
- CIFAR-100:      `CIFAR100_split_cn10_tn10_cet10_cs1_s2571.pkl`

If you do not have them yet, the helper scripts will regenerate them:

```bash
python scripts/generate_emnist_shuffle_split.py
python scripts/generate_cifar100_split.py
```

Raw data is downloaded automatically by `torchvision`.

## Results format

Each run writes to `./results/<algorithm>_<dataset>_<timestamp>/`:

- `results.json` &mdash; final / per-round / per-task accuracies and
  forgetting metrics
- `config.yaml`  &mdash; full resolved configuration
- `training.log` &mdash; full training log

Scripts under `scripts/` aggregate these JSONs into ablation summaries.

## Citation

If you find this code useful, please cite the corresponding paper:

```bibtex
@article{dyndfcl2026,
  title  = {Decentralized Federated Continual Learning with Dynamic Collaboration},
  author = {Li, Zirui and others},
  year   = {2026},
  note   = {Manuscript under review}
}
```

## License

[MIT](LICENSE).

## Acknowledgements

DynDFCL builds on the DCFCL backbone of Ma et al. (2024). The DER++
replay buffer follows Buzzega et al. (2020).
