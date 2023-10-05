# Plausibility of Attention Maps

by Duc Hau NGUYEN

## Abstract

This repository is used to implement various experimentations and gather results for the thesis of Duc Hau NGUYEN.

## TODO-list

- [ ] Visualization from different maps
- [ ] Morphosyntax post-filters: posterior filter on trained model `notebook`
- [ ] Morphosyntax pre-filters: filter during training `py script`
- [ ] Heuristics map: Precision-Recall curve and recompute AUPRC `notebook` 
- [ ] Visualization prediction from _json_ `py script`


## Structure

The repository is structured as follows:

```bash
├── README.md
├── requirements.cpu.txt
├── requirements.gpu.txt
├── template: template files for submitting jobs on the cluster
├── src: source code root folder
│   ├── data: pytorch Datasets 
│   ├── model: pytorch Modules
│   ├── data_module: pytorch LightningDataModule, store all data processing according to train/val/test/predict scenarios
│   ├── model_module: pytorch LightningModule, store all model processing logic
│   ├── module: helper modules
│   ├── *.py: training/inferring scripts
└── README.md 
```

## Dependencies

You'll need a working Python environment to run the code.

The recommended way to set up your environment is through the [Virtualenv Python](https://pypi.org/project/virtualenv/)
which provides the `virtual env`. The venv module supports creating lightweight “virtual environments”, each with their
own independent set of Python packages A virtual environment is created on top of an existing Python installation, known
as the virtual environment’s “base” Python, and may optionally be isolated from the packages in the base environment, so
only those explicitly installed in the virtual environment are available. (See further documentation
in [venv docs](https://docs.python.org/fr/3/library/venv.html)

The required dependencies are specified in the file `requirements.cpu.txt` (dependencies for CPU)
and `requirements.gpu.txt` (dependencies for GPU).

## Reproducing the results

1. Preparing dependencies for cpu from `requirements.gpu.txt`:

```bash
python -m venv eps
source eps/bin/activate
pip install -r requirements.cpu.words --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu113
```

2. Run an experimentation.
```bash
python src/single_lstm_attention_module.py \
          --cache .cache \
          --epoch 30 \
          --batch_size 128 \
          --vectors glove.840B.300d \
          --name hatexplain_supervise \
          --version run=2_lstm=1_lsup=0.6 \
          --data hatexplain \
          --lambda_supervise 0.6 \
          --n_lstm 1    
```

There 3 dataset for `--data`: `yelphat50`, `hatexplain`, `esnli`. 
Each technique has a corresponding lambda arguments:
  * **supervision**: `--lambda_supervise 0.5`
  * **regularization**: `--lambda_entropy 0.5`
  * **semisupervision**: `--lambda_heuristic 0.5` 
  * **regularization with guided entropy**: `--lambda_lagrange 0.5` 

3. Summarize results for figures. It will automatically create a new folder `summary` in `--out_dir` path.

```bash
python src/summarize_result.py \
          --log_dir .cache \ 
          --out_dir .cache \ 
          --figure 
          --experiment hatexplain_supervise
```

### Pip installations

```bash
pip install tables
```