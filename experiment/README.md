# Experiment: Custom Dataset Training

This experiment trains a transformer model on a custom dataset with a minimal character-level vocabulary.

## Dataset Specifications

- **Size**: ~350 million characters
- **Vocabulary**: Only 17 characters allowed:
  - Digits: `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`
  - Operators: `+`, `-`
  - Letters: `H`, `B`, `L`
  - Decimal: `.`
  - Newline: `\n`

## Training Budget Options

### $100 Budget (Current Default) — 2 GPUs

| Parameter | Value |
|-----------|-------|
| GPUs | **2× H100** |
| Depth | **24 layers** |
| Model dim | 1536 (24 × 64) |
| Vocab size | 32 |
| Total params | **~600M** |
| Device batch | 8 |
| Training time | **~16 hours** |
| Epochs | ~10 passes |
| Cost | 2 GPUs × 16h × $3 = **~$96** |

### $1000 Budget (Uncomment in config for full training) — 8 GPUs

| Parameter | Value |
|-----------|-------|
| GPUs | **8× H100** |
| Depth | **48-64 layers** |
| Model dim | 3072 (48 × 64) |
| Vocab size | 32 |
| Total params | **~2.4B** |
| Device batch | 4 |
| Training time | **~31 hours** |
| Epochs | ~70 passes |
| Cost | 8 GPUs × 31h × $3 = **~$744** |

## Model Configuration

Because the vocabulary is so small (32 tokens total including special tokens), the embedding tables are tiny. This frees up massive GPU memory that would normally be consumed by embeddings, allowing us to train a **much deeper model**.

| Parameter | Standard nanochat | This Experiment ($100) | This Experiment ($1000) |
|-----------|-------------------|------------------------|-------------------------|
| Depth | 32 layers | 24 layers | 48-64 layers |
| Vocab size | 65,536 | 32 | 32 |
| Embedding params | ~268M | ~49K | ~131K |
| Total params | ~1.88B | ~600M | ~2.4B |

With a small vocabulary, multiple passes (epochs) over the data are intentional—the model will deeply learn the patterns in your specialized domain.

## Quick Start

### 1. Prepare Your Data

Place your `.txt` files in:
```
~/.cache/nanochat_experiment/data/
```

**Format requirements:**
- Plain text files
- Only allowed characters: `0-9`, `+`, `-`, `H`, `B`, newline
- Separate documents with double newlines (`\n\n`)

Example file content:
```
123+456
789-123

HB+999-111
000+HB

456+789
```

### 2. Run Training

From the nanochat root directory:

```bash
# $100 budget training (~16 hours on 2×H100) - current default
bash experiment/run.sh

# With wandb logging
WANDB_RUN=my_experiment bash experiment/run.sh

# Override GPU count
NPROC_PER_NODE=8 bash experiment/run.sh  # 8 GPUs
NPROC_PER_NODE=1 bash experiment/run.sh  # Single GPU (slower)

# Customize depth (default: 24 for $100, 48 for $1000)
DEPTH=32 bash experiment/run.sh
```

### 3. Checkpoints

Checkpoints are saved to:
```
~/.cache/nanochat_experiment/checkpoints/d{depth}/
```

## Files

| File | Description |
|------|-------------|
| `tokenizer.py` | Character-level tokenizer (15 chars + 9 special = 24 tokens, padded to 32) |
| `dataset.py` | Dataset loading and validation |
| `dataloader.py` | Training data loader with DDP support |
| `train.py` | Main training script |
| `run.sh` | End-to-end training runner |

## Hyperparameters

### Model Architecture

| Depth | Params | VRAM | Recommended `device_batch_size` | Budget |
|-------|--------|------|--------------------------------|--------|
| 24 | ~600M | ~25GB | 8-12 | **$100** |
| 32 | ~1.0B | ~40GB | 6-8 | ~$300 |
| 48 | ~2.4B | ~60GB | 4-6 | **$1000** |
| 56 | ~2.8B | ~70GB | 3-4 | ~$1500 |
| 64 | ~3.2B | ~80GB | 2-3 | ~$2000 |

### Training

```bash
# $100 Budget (current default) - 2 GPUs
NPROC_PER_NODE=2            # Number of GPUs
DEPTH=24                    # Model depth
DEVICE_BATCH_SIZE=8         # Per-GPU batch size
TRAINING_HOURS=16.0         # Target training time (~16h on 2 GPUs)

# $1000 Budget (uncomment in run.sh) - 8 GPUs
# NPROC_PER_NODE=8          # Number of GPUs
# DEPTH=48                  # Model depth
# DEVICE_BATCH_SIZE=4       # Per-GPU batch size
# TRAINING_HOURS=31.0       # Target training time (~31h on 8 GPUs)
```

### Tuning Tips

1. **If OOM**: Reduce `DEVICE_BATCH_SIZE` (e.g., 4 → 2)
2. **If underutilizing GPU**: Increase depth or batch size
3. **For longer training**: Increase `TRAINING_HOURS`

## Memory Comparison

Standard nanochat with 65K vocab:
```
wte embedding:  65536 × 2048 = 134M params
lm_head:        65536 × 2048 = 134M params
Total embed:    268M params (~1GB in bf16)
```

This experiment with 32 vocab:
```
wte embedding:  32 × 3072 = 98K params
lm_head:        32 × 3072 = 98K params
Total embed:    196K params (~400KB in bf16)
```

**Savings: ~267M params (~1GB VRAM) freed for more transformer layers!**

## Custom Tokenizer Details

```python
# Character mappings (IDs 0-16)
'0' -> 0    '1' -> 1    '2' -> 2    '3' -> 3    '4' -> 4
'5' -> 5    '6' -> 6    '7' -> 7    '8' -> 8    '9' -> 9
'+' -> 10   '-' -> 11   'H' -> 12   'B' -> 13   'L' -> 14
'.' -> 15   '\n' -> 16

# Special tokens (ID 17)
<|bos|> -> 17

# Padding (IDs 18-31) - unused, for alignment
```

## Notes

- **Many epochs**: With 350M tokens and ~25B tokens processed, expect ~70 epochs
- **Overfitting risk**: Monitor validation loss; may plateau early
- **Specialized domain**: This setup is designed for pattern learning in structured data
- **No BPE**: Character-level tokenization means 1 token = 1 character
