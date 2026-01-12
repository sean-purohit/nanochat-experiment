#!/bin/bash

# =============================================================================
# Experiment Training Script - Custom Dataset with Character-Level Tokenizer
# =============================================================================
#
# This script trains a transformer model on your custom dataset with a minimal 
# vocabulary (15 characters + special tokens).
#
# =============================================================================
# BUDGET CONFIGURATIONS:
# =============================================================================
#
# ---- $100 BUDGET (CURRENT) ----
# Budget: ~$100 (~4 hours on 8xH100 at $3/GPU/hr)
# Model depth: 24 layers
# Model dim: 1536 (24 * 64 aspect ratio)
# Parameters: ~600M
# Training time: ~4 hours
# Epochs over data: ~10 (moderate passes)
#
# ---- $1000 BUDGET (UNCOMMENT FOR FULL TRAINING) ----
# Budget: ~$1000 (~41.6 hours on 8xH100 at $3/GPU/hr)
# Model depth: 48 layers (vs 32 for standard $1000 run)
# Model dim: 3072 (48 * 64 aspect ratio)
# Parameters: ~2.4B
# Training time: ~31 hours (pretraining)
# Epochs over data: ~70 (many passes due to small dataset)
#
# The model is DEEPER than standard nanochat because:
# 1. Tiny vocab = tiny embedding tables = more room for transformer layers
# 2. Deep models are better at learning complex patterns
#
# =============================================================================

set -e  # Exit on error

# Configuration
export OMP_NUM_THREADS=1
export NANOCHAT_EXPERIMENT_DIR="$HOME/.cache/nanochat_experiment"
mkdir -p $NANOCHAT_EXPERIMENT_DIR
mkdir -p $NANOCHAT_EXPERIMENT_DIR/data
mkdir -p $NANOCHAT_EXPERIMENT_DIR/checkpoints

# =============================================================================
# GPU CONFIGURATION
# =============================================================================
# Number of GPUs (2 for $100 budget, 8 for $1000 budget)
NPROC_PER_NODE=${NPROC_PER_NODE:-2}

# # For $1000 budget with 8 GPUs (uncomment later):
# NPROC_PER_NODE=${NPROC_PER_NODE:-8}

# wandb run name (set to "dummy" to disable)
WANDB_RUN=${WANDB_RUN:-dummy}

# =============================================================================
# $100 BUDGET CONFIGURATION (CURRENT) - 2 GPUs
# =============================================================================
# Model depth: d24 = ~600M params, uses ~25GB VRAM with batch_size=8
DEPTH=${DEPTH:-24}

# Device batch size - larger for smaller models
DEVICE_BATCH_SIZE=${DEVICE_BATCH_SIZE:-8}

# Training duration: ~16 hours for $100 budget (2 GPUs × 16h × $3/GPU = $96)
# Same compute as 8 GPUs × 4h, just spread across fewer GPUs
TRAINING_HOURS=${TRAINING_HOURS:-16.0}

# =============================================================================
# $1000 BUDGET CONFIGURATION (UNCOMMENT FOR FULL TRAINING) - 8 GPUs
# =============================================================================
# # Model depth - DEEP due to tiny vocab
# # d48 = ~2.4B params, uses ~60GB VRAM with batch_size=4
# # d64 = ~3.2B params, might need batch_size=2
# DEPTH=${DEPTH:-48}
#
# # Device batch size - smaller for deeper models
# # d48: use 4-6
# # d64: use 2-4
# DEVICE_BATCH_SIZE=${DEVICE_BATCH_SIZE:-4}
#
# # Training duration in hours (set to fill $1000 budget with 8 GPUs)
# TRAINING_HOURS=${TRAINING_HOURS:-31.0}

echo "============================================================"
echo "EXPERIMENT: Custom Dataset Training"
echo "============================================================"
echo "Data directory: $NANOCHAT_EXPERIMENT_DIR/data"
echo "Checkpoints: $NANOCHAT_EXPERIMENT_DIR/checkpoints"
echo "Model depth: $DEPTH"
echo "Device batch size: $DEVICE_BATCH_SIZE"
echo "Training hours: $TRAINING_HOURS"
echo "GPUs: $NPROC_PER_NODE"
echo "wandb: $WANDB_RUN"
echo "============================================================"
echo ""

# -----------------------------------------------------------------------------
# Step 0: Setup Python environment
# -----------------------------------------------------------------------------

echo "[Step 0] Setting up Python environment..."

# Install uv if not present
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv if needed
[ -d ".venv" ] || uv venv

# Install dependencies
uv sync --extra gpu

# Activate venv
source .venv/bin/activate

echo "Python environment ready."
echo ""

# -----------------------------------------------------------------------------
# Step 1: Validate data
# -----------------------------------------------------------------------------

echo "[Step 1] Validating dataset..."

DATA_DIR="$NANOCHAT_EXPERIMENT_DIR/data"

# Check if data exists
if [ -z "$(ls -A $DATA_DIR/*.txt 2>/dev/null)" ]; then
    echo ""
    echo "ERROR: No data files found!"
    echo ""
    echo "Please place your training data in: $DATA_DIR"
    echo ""
    echo "Data format:"
    echo "  - Plain text files (.txt)"
    echo "  - Only allowed characters: 0-9, +, -, H, B, newline"
    echo "  - Separate documents with double newlines"
    echo ""
    echo "Example:"
    echo "  123+456"
    echo "  789-123"
    echo "  HB+999"
    echo ""
    exit 1
fi

# Validate the data
python -m experiment.dataset --validate

echo "Dataset validation passed!"
echo ""

# -----------------------------------------------------------------------------
# Step 2: Test tokenizer
# -----------------------------------------------------------------------------

echo "[Step 2] Testing tokenizer..."

python -m experiment.tokenizer

echo "Tokenizer ready!"
echo ""

# -----------------------------------------------------------------------------
# Step 3: Train the model
# -----------------------------------------------------------------------------

echo "[Step 3] Starting training..."
echo ""
echo "Model configuration:"
echo "  - Depth: $DEPTH layers"
echo "  - Estimated params: ~$(python -c "print(f'{$DEPTH * 50.4:.1f}M')")"
echo "  - Vocab size: 32 (15 chars + 9 special + 8 padding)"
echo "  - Device batch size: $DEVICE_BATCH_SIZE"
echo "  - Training hours: $TRAINING_HOURS"
echo ""

if [ "$NPROC_PER_NODE" -gt 1 ]; then
    echo "Running distributed training on $NPROC_PER_NODE GPUs..."
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m experiment.train \
        --depth=$DEPTH \
        --device_batch_size=$DEVICE_BATCH_SIZE \
        --training_hours=$TRAINING_HOURS \
        --run=$WANDB_RUN
else
    echo "Running single-GPU training..."
    python -m experiment.train \
        --depth=$DEPTH \
        --device_batch_size=$DEVICE_BATCH_SIZE \
        --training_hours=$TRAINING_HOURS \
        --run=$WANDB_RUN
fi

echo ""
echo "============================================================"
echo "Training complete!"
echo "============================================================"
echo ""
echo "Checkpoints saved to: $NANOCHAT_EXPERIMENT_DIR/checkpoints/d$DEPTH/"
echo ""
echo "To chat with your model:"
echo "  python -m experiment.chat"
echo ""
