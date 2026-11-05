<div align="center">

# ClinStab

### Stability-Oriented Learning for Medical Time Series via Dual-Stream Alignment

<strong>Wenshuo Chen<sup>†</sup>, Haoxuan Xu<sup>†</sup>, Yi Liu, Songning Lai, Ninghui Feng, Jiayu Yang, Haoang Li<sup>*</sup>, Yutao Yue<sup>*</sup></strong><br />
<sup>†</sup>Equal contribution · <sup>*</sup>Corresponding authors

[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-7b2cbf.svg)](https://neurips.cc/Conferences/2026)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.0-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-2ea44f.svg)](LICENSE)

**ClinStab is a Medformer-based framework for stability-oriented EEG and ECG classification.**
It combines Salience-Context Dual Modulation (SCDM) with perturbation-aware training to study the clean/robustness tradeoff under the evaluated stress tests.

<p>
  <a href="#quick-start">Quick start</a> ·
  <a href="#data-preparation">Data</a> ·
  <a href="#running-the-model">Run the model</a> ·
  <a href="#citation">Citation</a>
</p>

</div>

<p align="center">
  <img src="assets/clinstab_framework.png" alt="ClinStab framework" width="920" />
</p>

## Overview

ClinStab keeps the Medformer backbone and replaces its interaction layer with a channel-aware dual-stream module. SCDM uses channel energy as a routing prior: the selected high-energy components are processed by sparse attention, while the remaining components are processed by a lightweight depth-wise context pathway and restored to their original channel order. The training objective combines clean supervision, perturbed supervision, prediction consistency, and teacher-guided stabilization.

The release contains the runnable ClinStab model, stability-objective utilities, Medformer-based training entry points, and lightweight verification tests.

## At a glance

| Item | Configuration |
| --- | --- |
| Backbone | Medformer-style multi-granularity encoder |
| Modalities | EEG and ECG |
| Benchmarks | APAVA, TDBRAIN, ADFTD, PTB |
| Routing | Hard energy TopK with `k=floor(C / alpha)` |
| Default patch lengths | `2,4,8` |
| Reference environment | Python 3.10 · PyTorch 2.0.0 · CUDA 11.8 |

## Quick start

```bash
git clone https://github.com/xll0328/NeurIPS26-ClinStb.git
cd NeurIPS26-ClinStb

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu118 -r requirements.txt
```

Run the model and objective smoke tests before preparing a dataset:

```bash
python -m unittest discover -s tests -v
```

For a CPU-only setup, install the PyTorch build matching your platform instead of the CUDA 11.8 wheel.

## Data preparation

The loaders expect processed arrays in the following layout:

```text
dataset/
└── DATA_NAME/
    ├── Feature/
    │   ├── feature_0.npy
    │   ├── feature_1.npy
    │   └── ...
    └── Label/
        └── label.npy
```

Each feature file stores one subject's samples with shape `[num_samples, sequence_length, channels]`. The label file stores the corresponding subject labels and IDs. Obtain the datasets from their providers and follow their access terms:

- [APAVA](https://osf.io/jbysn/)
- [TDBRAIN](https://brainclinics.com/resources/)
- [ADFTD](https://openneuro.org/datasets/ds004504/versions/1.0.6)
- [PTB](https://physionet.org/content/ptbdb/1.0.0/)

## Running the model

### Clean classification

After preparing APAVA under `dataset/APAVA/`, run:

```bash
python -u run.py \
  --task_name classification --is_training 1 \
  --root_path ./dataset/APAVA/ --model_id APAVA-Indep \
  --model ClinStab --data APAVA \
  --e_layers 6 --batch_size 64 --d_model 128 --d_ff 256 \
  --n_heads 8 --patch_len_list 2,4,8 --scdm_alpha 4.67 \
  --augmentations flip,frequency,jitter,mask,channel,drop \
  --des ClinStab --itr 5 --learning_rate 0.0001 \
  --train_epochs 100 --patience 10
```

The `--scdm_alpha` argument controls the attention budget at each granularity. The model sends `floor(channels / scdm_alpha)` channels to the attention stream and keeps the other channels in the context stream.

### Stability-oriented training

Run the clean command first so that the corresponding checkpoint is available, then launch the robustness stage with the same data and model settings:

```bash
python -u run_rob.py \
  --task_name classification --is_training 1 \
  --root_path ./dataset/APAVA/ --model_id APAVA-Indep \
  --model ClinStab --data APAVA \
  --e_layers 6 --batch_size 64 --d_model 128 --d_ff 256 \
  --n_heads 8 --patch_len_list 2,4,8 --scdm_alpha 4.67 \
  --augmentations flip,frequency,jitter,mask,channel,drop \
  --des ClinStab --itr 5 --learning_rate 0.0001 \
  --train_epochs 100 --patience 10 \
  --eps 0.20 --steps 10 --rho 0.65 --max_restarts 1
```

Use `python run_rob.py --help` to inspect the complete set of perturbation and optimization options.

### Standalone objective utilities

The model-independent helpers in `utils/clinstab_objective.py` can be used with a custom training loop:

```python
from utils.clinstab_objective import projected_gradient_attack, stability_objective

perturbed = projected_gradient_attack(
    model, x, y, epsilon=0.20, step_size=0.01, steps=10
)
clean_logits = model(x)
perturbed_logits = model(perturbed)
# Set this to frozen teacher logits when a separate clean teacher is available.
teacher_logits = None
loss, terms = stability_objective(
    clean_logits,
    perturbed_logits,
    y,
    teacher_logits=teacher_logits,
    lambda_rob=1.0,
    lambda_con=0.5,
    lambda_know=0.5,
    temperature=2.0,
)
```

## Repository layout

```text
.
├── models/ClinStab.py              # SCDM model
├── utils/clinstab_objective.py     # bounded PGD and stability losses
├── run.py                          # clean classification entry point
├── run_rob.py                      # stability-training entry point
├── data_provider/                  # processed dataset loaders
├── tests/                          # model and objective smoke tests
├── assets/                         # README method figure
├── requirements.txt
└── LICENSE
```

## Implementation notes

- Seeds `41–45` are used by the provided multi-run commands.
- Each sample is standardized per channel by the shared `normalize_batch_ts` preprocessing helper.
- The SCDM routing indices are recomputed per sample and granularity during the forward pass.
- The base data and training framework is adapted from the [Medformer repository](https://github.com/DL4mHealth/Medformer/tree/446275f27b713a9f09917a6ba0bc51a18e921597); upstream notices are retained.

## Citation

Wenshuo Chen and Haoxuan Xu contributed equally to this work.
Haoang Li and Yutao Yue are corresponding authors.

```bibtex
@inproceedings{clinstab2026,
  title={ClinStab: Stability-Oriented Learning for Medical Time Series via Dual-Stream Alignment},
  author={Chen, Wenshuo and Xu, Haoxuan and Liu, Yi and Lai, Songning and Feng, Ninghui and Yang, Jiayu and Li, Haoang and Yue, Yutao},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```

## License

The released code is provided under the [MIT License](LICENSE). Dataset access and any upstream components remain subject to their respective provider and license terms.

Questions and bug reports are welcome through [GitHub Issues](https://github.com/xll0328/NeurIPS26-ClinStb/issues).
