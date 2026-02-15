# SpecTemp

**Spectral Tempering for Embedding Compression in Dense Passage Retrieval**

SpecTemp is a post-hoc embedding compression method that applies spectral tempering to reduce the dimensionality of dense retrieval embeddings while preserving retrieval performance.

---

## Datasets

Experiments use datasets from the [Extended BEIR Datasets](https://github.com/liyongkang123/extended_beir_datasets) repository. Datasets are stored in the `datasets/` folder and will be downloaded automatically when the code is run.

---

## Environment Setup

You can set up the environment using either pip or conda.

```bash
# Using pip (recommended)
pip install -r requirements.txt

# Using conda
conda env create -f environment.yml
conda activate <env_name>
```

Beyond standard deep learning libraries (PyTorch, Transformers), this project requires:

- [BEIR](https://github.com/beir-cellar/beir) — Benchmarking IR library

---

## Usage

Evaluation is a two-step process: first save the embeddings, then run compression.

### Step 1 — Save Embeddings

Run dense retrieval and save the generated embeddings to disk. This only needs to be done once per model/dataset combination.

**Run all models and datasets via SLURM:**
```bash
bash scripts/eval_embedding_save.sh
```

**Run a single model manually:**
```bash
python eval_emebedding_save.py \
    --model_name qwen3 \
    --dataset msmarco \
    --per_gpu_eval_batch_size 32
```

Supported `--model_name` values: `gte`, `qwen3`, `jina_v4`, `bge_m3`, `nomic_v2`, `embeddinggemma`

---

### Step 2 — Evaluate Embedding Compression

Load the saved embeddings, apply a compression transform, and evaluate retrieval performance.

**Run all configurations via SLURM:**
```bash
# First run the "none" baseline (required for paired t-tests in subsequent runs)
bash scripts/eval_embedding_compression.sh
```

**Run a single configuration manually:**
```bash
python eval_embedding_compression.py \
    --model_name qwen3 \
    --dataset msmarco \
    --transform_type spectemp \
    --target_dim 768 \
    --seed 2026
```

Supported `--transform_type` values:

| Transform | Description |
|-----------|-------------|
| `none` | No compression (baseline) |
| `prefix_truncation` | Matryoshka-style prefix truncation |
| `random_truncation` | Random dimension truncation |
| `random_projection` | Random projection |
| `pca` | PCA-based truncation |
| `whitening` | Whitening + truncation |
| `y-whitening` | Y-whitening (γ = 0.5) |
| `spectemp` | Spectral Tempering (ours) |

> **Note:** The `none` baseline must be run before any other transform, as it is used as the reference for paired t-tests.
