# Supervision, Semi-supervision and Regularization to Improve the Plausibility of Attention in RNN Encoders

by Anonymous authors

## Abstract

Many recent advances in machine learning for natural language processing rely on the attention mechanism. This mechanism
results in an attention map that highlights the most relevant words, or tokens, that prompt a certain decision for the
model. While many empirical studies on a few examples postulate that attention maps can provide a justification for how
the decision was made, only a few look for making this map understandable to humans, i.e., improving the plausibility of
attention maps. Recent experiments with annotated data show that brute attention weights are hardly plausible because
they are too spread on the input tokens. We thus explore several possibilities to improve models' plausibility without
changing their architecture. In particular, one can think of regularization to increase the sparsity of attention
weights, and supervision of the attention weights with either reference annotations or automatically generated ones. In
this work, we study the impact of these techniques on plausibility and their effect on the model's performance.
Supervision and regularization are cast as additional constraints to the learning objectives that apply to the attention
layer of a bi-LSTM encoder. Results in natural language inference (NLI) show that regularization is a way to increase
plausibility, but the same experiment on sentiment classification and hate speech classification tasks does not yield
the same increase, showing that this method is task-dependent. Also, the particular instruction for annotation on NLI
worsens the classification performance, which is not the case in other tasks. Beyond the attention map, the result of
experiments on text classification tasks also shows that no matter how the gain is brought by the constraint, the
contextualization layer plays a crucial role in finding the right space for finding plausible tokens.

## Software implementation

Experimentations are produced by `src/lstm_attention.py`, then average performances are synthetized by
`src/summarize_result.py`. Source code used to generate the results and figures in the paper are in the `notebooks`
folder.

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

## Citations

Anonymous

## Contributing

Anonymous