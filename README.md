# ClinStab

ClinStab is a Medformer-based medical time-series classifier with the Salience-Context Dual Modulation (SCDM) block and stability-oriented training. The implementation in `models/ClinStab.py` follows the paper specification: multi-granularity channel-aligned patching, energy-based hard TopK routing, attention for the selected channels, depth-wise temporal convolution for the residual channels, channel-order-preserving fusion, and residual classification.

## Setup

The reference environment is Python 3.10, PyTorch 2.0.0 with CUDA 11.8, and the packages in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu118 -r requirements.txt
```

For a CPU-only installation, replace the PyTorch line in `requirements.txt` with the compatible CPU build for the local system.

## Dataset layout

The code expects processed data under `dataset/DATA_NAME/`:

```text
dataset/DATA_NAME/
├── Feature/feature_ID.npy
└── Label/label.npy
```

Each `feature_ID.npy` stores one subject's samples with shape `[number_of_samples, sequence_length, channels]`; `Label/label.npy` stores subject labels and IDs. The datasets are not bundled. Obtain them from [APAVA](https://osf.io/jbysn/), [ADFTD](https://openneuro.org/datasets/ds004504/versions/1.0.6), [PTB](https://physionet.org/content/ptbdb/1.0.0/), [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/), or [TDBRAIN](https://brainclinics.com/resources/) and follow the providers' access terms.

## ClinStab clean training

After preparing APAVA, run the paper model with:

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

`--scdm_alpha` controls the hard routing budget: each granularity sends `floor(channels / scdm_alpha)` channels to the attention stream and keeps all remaining channels in the context stream. The output of the model is a class-logit tensor of shape `[batch, num_classes]`.

## Stability training utilities

`utils/clinstab_objective.py` provides the paper's bounded perturbation helper and composite objective:

```python
from utils.clinstab_objective import projected_gradient_attack, stability_objective

perturbed = projected_gradient_attack(model, x, y, epsilon=0.20, step_size=0.01, steps=10)
clean_logits = model(x)
perturbed_logits = model(perturbed)
loss, terms = stability_objective(
    clean_logits, perturbed_logits, y, teacher_logits=teacher_logits,
    lambda_rob=1.0, lambda_con=0.5, lambda_know=0.5, temperature=2.0,
)
```

The existing `run_rob.py` entry point can be used for the robustness-training workflow with `--model ClinStab` and the perturbation options documented by `python run_rob.py --help`.

Before using a dataset, run the lightweight implementation checks:

```bash
python -m unittest discover -s tests -v
```

## Code layout

- `models/ClinStab.py`: SCDM model and channel-preserving multi-granularity encoder.
- `utils/clinstab_objective.py`: bounded PGD perturbation and clean/robust/consistency/teacher-KL losses.
- `run.py`: clean classification training/evaluation entry point.
- `run_rob.py`: robustness-training entry point.
- `data_provider/`: loaders for the processed benchmark layout.
- `tests/`: model, routing, gradient, and stability-objective smoke tests.

The release provides the runnable model, objective utilities, training entry points, and lightweight verification tests. Dataset files are obtained separately from the providers listed below.

## Upstream implementation

The base data and training framework builds on the public [Medformer repository](https://github.com/DL4mHealth/Medformer/tree/446275f27b713a9f09917a6ba0bc51a18e921597). The included `LICENSE` file applies to the released code and upstream notices are retained.

## Citation

```bibtex
@inproceedings{clinstab2026,
  title={ClinStab: Stability-Oriented Learning for Medical Time Series via Dual-Stream Alignment},
  author={Chen, Wenshuo and Xu, Haoxuan and Liu, Yi and Lai, Songning and Feng, Ninghui and Yang, Jiayu and Li, Haoang and Yue, Yutao},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```
