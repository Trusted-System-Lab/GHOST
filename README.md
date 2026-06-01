# GHOST 

Official repo for the paper "Mitigating Gradient Inversion Risks in Language Models via Token Obfuscation" in Asia CCS'26.

This is a reconstruction of the original GHOST code base. Some of the experimental results might differ.

## Installation

This repo uses `uv` for dependency management. Install `uv` first if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then create the environment and install all dependencies from the lockfile:

```bash
uv sync
```

The project pins the main experimental dependencies in `pyproject.toml`, including PyTorch, Transformers, Datasets, Evaluate, Accelerate, spaCy, NLTK, PEFT, Flair, Rouge, and scikit-learn. The spaCy English model is also installed through `uv`.

Download the NLTK tokenizer data used by the preprocessing scripts:

```bash
uv run python -c "import nltk; nltk.download('punkt')"
```

For Linux/CUDA runs with Llama or Gemma 4-bit loading, `bitsandbytes` is installed automatically. It is skipped on macOS because official `bitsandbytes==0.45.0` wheels are not available for macOS arm64.

If you use gated Hugging Face models such as Llama or Gemma, set an access token:

```bash
export HF_TOKEN=your_huggingface_token
```

## Usage

Run the BERT-style classification pipeline with:

```bash
uv run python -m ghost.cli --data sst2 --model_name bert-base-uncased --num_of_samples 1000
```

Run the decoder/generation pipeline with GPT-2:

```bash
uv run ghost-transform --task generation --data enron --model_name gpt2 --num_of_samples 1000
```

For gated Llama/Gemma models, provide an HF token via `HF_TOKEN` or `--hf_token`.
Large decoder models can be loaded with 4-bit quantization on Linux/CUDA:

```bash
HF_TOKEN=... uv run ghost-transform \
  --task generation \
  --data enron \
  --model_name meta-llama/Llama-2-7b-hf \
  --device cuda:0 \
  --load_in_4bit \
  --add_eos_token
```

The pipeline has two explicit stages:

1. `ghost.search.ShadowTokenSearcher` builds per-token shadow candidate sets from
   embedding neighbors, then filters indirect similarity, direct mutual-neighbor
   similarity, and common-lemma similarity.
2. `ghost.select.HiddenStateSelector` performs coordinate beam search over those
   candidates to minimize hidden-state MSE against the original tokenized sentence.

Outputs and shadow-token caches are written under `data/<model-name>/` by default.
