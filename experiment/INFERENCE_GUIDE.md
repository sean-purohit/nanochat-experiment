# Inference Guide

This guide explains how to use your trained models for inference.

## Quick Start

### 1. Single Prompt Generation

Generate text from a simple prompt:

```bash
cd /Users/sot/Documents/26TBOT/nanochat
python -m experiment.inference --checkpoint step_002920 --prompt "123+456="
```

### 2. Interactive Mode

Run in interactive mode for multiple prompts:

```bash
python -m experiment.inference --checkpoint step_002920 --interactive
```

In interactive mode, you can:
- Enter prompts and get generations
- Use `/temp 0.5` to change temperature
- Use `/topk 10` to use top-k sampling
- Use `/tokens 50` to change max tokens
- Use `/quit` to exit

## Available Checkpoints

Your trained models are in `experiment/checkpoints/`:

| Checkpoint | Val Loss | Best For | Size |
|------------|----------|----------|------|
| **step_002920** | **0.2627** | ⭐ **Best overall** | 2.5GB |
| step_003650 | 0.2624 | Excellent, slightly past optimal | 5.3GB |
| step_004380 | 0.2666 | Good, slight overfitting | 5.3GB |

**Recommendation:** Use `step_002920` for best results!

## Command Line Options

```bash
python -m experiment.inference [OPTIONS]

Required:
  --checkpoint FOLDER       Checkpoint folder name (e.g., step_002920)

Optional:
  --prompt TEXT            Prompt for generation
  --max_tokens N           Maximum tokens to generate (default: 100)
  --temperature T          Sampling temperature 0.0-1.0 (default: 0.8)
                           - 0.0 = greedy (always pick most likely)
                           - 1.0 = random (sample from full distribution)
  --top_k N               Top-k sampling (default: None)
                           - Only sample from top N most likely tokens
  --interactive            Run in interactive mode
  --device cpu/cuda/mps    Device to use (default: auto-detect)
```

## Examples

### Example 1: Simple Math

```bash
python -m experiment.inference \
    --checkpoint step_002920 \
    --prompt "123+456=" \
    --max_tokens 20
```

### Example 2: Greedy Decoding

Use temperature 0.0 for deterministic output:

```bash
python -m experiment.inference \
    --checkpoint step_002920 \
    --prompt "H+" \
    --temperature 0.0 \
    --max_tokens 50
```

### Example 3: Creative Sampling

Use higher temperature and top-k for more variety:

```bash
python -m experiment.inference \
    --checkpoint step_002920 \
    --prompt "B-" \
    --temperature 1.2 \
    --top_k 10 \
    --max_tokens 100
```

### Example 4: Compare Checkpoints

Compare outputs from different checkpoints:

```bash
# Best checkpoint
python -m experiment.inference --checkpoint step_002920 --prompt "123+456="

# Later checkpoint (more training)
python -m experiment.inference --checkpoint step_003650 --prompt "123+456="

# Overfitted checkpoint
python -m experiment.inference --checkpoint step_004380 --prompt "123+456="
```

## Understanding Your Tokenizer

Your model uses a **character-level tokenizer** with this vocabulary:

**Characters (15 tokens):**
- Digits: `0123456789` (10 tokens)
- Operators: `+-` (2 tokens)
- Letters: `HBL` (3 tokens)
- Decimal: `.` (1 token)
- Newline: `\n` (1 token)

**Special tokens (1 token):**
- `<|bos|>` - Beginning of sequence

**Total vocab size:** 32 (padded for efficiency)

### What This Means:
- ✅ Your model can generate: `0-9`, `+`, `-`, `H`, `B`, `L`, `.`, `\n`
- ❌ Any other characters will be skipped
- 💡 Model was trained on your specific dataset format

## Tips for Best Results

### Temperature Guide:
- **0.0 - 0.3**: Very focused, deterministic (good for math)
- **0.5 - 0.8**: Balanced (recommended default)
- **0.9 - 1.5**: Creative, random (good for exploration)

### Top-K Guide:
- **None**: Use full probability distribution
- **5-10**: Somewhat constrained
- **20-40**: Fairly diverse
- **Higher**: More random

### When to Use Each Checkpoint:

**Use step_002920 when:**
- ✅ You want the most accurate predictions
- ✅ You need best generalization
- ✅ Input is similar but not identical to training data

**Use step_003650 when:**
- ✅ As a backup to step_002920
- ✅ Comparing outputs

**Use step_004380 when:**
- ⚠️ For comparison only (slight overfitting)

## Troubleshooting

### "Checkpoint directory not found"
- Check that checkpoint folder exists in `experiment/checkpoints/`
- Use exact folder name: `step_002920` not `002920`

### "No model files found"
- Ensure `model_XXXXXX.pt` file exists in checkpoint folder
- Metadata file `meta_XXXXXX.json` should also exist

### Slow generation on CPU
- Use `--device cuda` if you have a GPU
- Use `--device mps` if you have Apple Silicon Mac
- Reduce `--max_tokens` for faster generation

### Out of memory
- Use smaller checkpoint (step_002920 = 2.5GB)
- Reduce `--max_tokens`
- Use `--device cpu` (slower but more memory)

## Interactive Mode Commands

When running with `--interactive`:

| Command | Description | Example |
|---------|-------------|---------|
| `/temp <value>` | Set temperature | `/temp 0.5` |
| `/topk <value>` | Set top-k | `/topk 10` |
| `/tokens <n>` | Set max tokens | `/tokens 50` |
| `/quit` | Exit | `/quit` |

Just type your prompt and press Enter to generate!

## Next Steps

After testing inference:

1. **Evaluate on test data** - See how well it performs
2. **Experiment with sampling** - Try different temperatures
3. **Compare checkpoints** - Validate that step_002920 is best
4. **Document findings** - Note what works well

## Questions?

Check the model metadata:
```bash
cat experiment/checkpoints/step_002920/meta_002920.json | python -m json.tool
```

Happy generating! 🎉
