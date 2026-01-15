# $1000 Training Setup Guide
## 8x H100 PCIe GPUs - Maximum Depth Configuration

This guide explains the setup for training the **4.3B parameter model** (depth 64) on 8x H100 PCIe GPUs.

---

## 🖥️ Hardware Specifications

| Component | Specification |
|-----------|--------------|
| **GPUs** | 8x H100 PCIe (80GB each) |
| **Total VRAM** | 640 GB |
| **RAM** | 1408 GB |
| **vCPU** | 248 cores |
| **Disk** | 270 GB (ephemeral) |
| **Persistent Storage** | /workspace (network volume) |

---

## 📊 Model Configuration

### **Maximum Depth: d64**

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Depth** | 64 layers | Maximum for 80GB H100 |
| **Model Dim** | 4096 (64 × 64) | aspect_ratio=64 |
| **Parameters** | ~4.3B | vs 680M for d24 |
| **VRAM per GPU** | ~75GB | With batch_size=4 |
| **Heads** | 32 | head_dim=128 |
| **KV Heads** | 32 (MHA) | Can use GQA if needed |
| **Vocab Size** | 32 | Character-level |
| **Sequence Length** | 2048 | tokens |

### **Why Depth 64?**

```
Memory Budget per GPU: 80GB

d48 (2.4B params):  ~60GB ✅ (safe, underutilized)
d64 (4.3B params):  ~75GB ✅ (optimal, 5GB headroom)
d80 (6.7B params):  ~95GB ❌ (OOM risk)

Choice: d64 - Maximum depth with safe margins
```

---

## 💰 Budget Breakdown

**Target: $1000**

| Item | Calculation | Cost |
|------|-------------|------|
| GPU Cost | $4/GPU/hour | - |
| Total GPUs | 8 GPUs | - |
| Training Hours | 31 hours | - |
| **Total** | 8 × 31 × $4 | **$992** |

**Training Schedule:**
- Start training
- Runs for 31 hours
- Auto-stops at budget limit
- Checkpoints saved every hour

---

## 🚀 Setup Instructions

### **Step 1: Connect to RunPod**

```bash
# You will receive new connection info
ssh root@<NEW_IP> -p <NEW_PORT> -i ~/.ssh/id_ed25519
```

### **Step 2: Verify Hardware**

```bash
# Check GPUs
nvidia-smi

# Should show:
# - 8x NVIDIA H100 PCIe
# - 80GB memory each
# - All GPUs idle and ready

# Check disk space
df -h /workspace
# Should show large network volume (not 270GB local)
```

### **Step 3: Locate /workspace**

```bash
# Check workspace location
cd /workspace
pwd
ls -la

# Workspace should be:
# - Network-mounted persistent storage
# - Large capacity (not the 270GB local disk)
# - Survives pod restarts
```

### **Step 4: Clone Repository**

```bash
cd /workspace
git clone https://github.com/sean-purohit/nanochat-experiment.git
cd nanochat-experiment
```

### **Step 5: Upload Dataset**

```bash
# From your local machine:
scp -P <PORT> -i ~/.ssh/id_ed25519 \
    nanochat/experiment/dataset/combined_output.txt \
    root@<IP>:/workspace/nanochat-experiment/experiment/dataset/
```

### **Step 6: Install Dependencies**

```bash
# RunPod PyTorch template should have most deps
# If needed:
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### **Step 7: Verify Configuration**

```bash
cd /workspace/nanochat-experiment

# Check training config
grep "DEPTH=" experiment/run.sh
# Should show: DEPTH=${DEPTH:-64}

grep "NPROC_PER_NODE=" experiment/run.sh
# Should show: NPROC_PER_NODE=${NPROC_PER_NODE:-8}

grep "TRAINING_HOURS=" experiment/run.sh
# Should show: TRAINING_HOURS=${TRAINING_HOURS:-31.0}
```

### **Step 8: Test Environment**

```bash
# Test PyTorch CUDA
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Should output:
# CUDA available: True
# GPU count: 8
```

---

## 🎯 Starting Training

### **In tmux (Recommended)**

```bash
# Create persistent session
tmux new-session -s train_d64

# Navigate to repo
cd /workspace/nanochat-experiment

# Start training
bash experiment/run.sh
```

**Detach from tmux:** `Ctrl+B`, then `D`  
**Reattach:** `tmux attach -t train_d64`

### **Training Command Breakdown**

The `run.sh` script will execute:

```bash
torchrun --standalone --nproc_per_node=8 \
    -m experiment.train \
    --depth=64 \
    --device_batch_size=4 \
    --training_hours=31.0 \
    --save_every=730 \
    --wandb_run=dummy
```

**Parameters:**
- `--nproc_per_node=8`: Use all 8 GPUs
- `--depth=64`: 4.3B parameter model
- `--device_batch_size=4`: 4 samples per GPU (~75GB VRAM)
- `--training_hours=31.0`: Fill $1000 budget
- `--save_every=730`: Checkpoint every hour
- Total batch size: 8 GPUs × 4 batch × 2048 seq = 524,288 tokens

---

## 📦 Checkpoint Management

### **Storage Location**

```bash
/workspace/nanochat_experiment/checkpoints/d64/
```

**Files per checkpoint (~17GB each):**
- `model_XXXXXX.pt` (10.7GB) - Model weights
- `optim_XXXXXX_rank0.pt` through `rank7.pt` (8 × 0.8GB) - Optimizer states
- `meta_XXXXXX.json` (1KB) - Metadata

**Total Storage:**
- Checkpoints per hour: 1
- Total hours: 31
- Checkpoint size: ~17GB
- **Total**: ~527GB (fits in workspace)

### **Checkpoint Schedule**

| Hour | Step | Checkpoint | Cumulative Size |
|------|------|------------|-----------------|
| 1 | 730 | ✅ | 17GB |
| 2 | 1460 | ✅ | 34GB |
| ... | ... | ... | ... |
| 31 | 22,630 | ✅ | ~527GB |

### **Download Best Checkpoint**

```bash
# From local machine
scp -P <PORT> -i ~/.ssh/id_ed25519 -r \
    root@<IP>:/workspace/nanochat_experiment/checkpoints/d64/model_*.pt \
    ./local_checkpoints/
```

---

## 📈 Monitoring

### **TensorBoard**

```bash
# On RunPod server
tmux new-session -s tensorboard
tensorboard --logdir=/workspace/nanochat_experiment/tensorboard_logs --host=0.0.0.0 --port=6006

# On local machine (SSH tunnel)
ssh -L 6006:localhost:6006 root@<IP> -p <PORT> -i ~/.ssh/id_ed25519

# Open: http://localhost:6006
```

### **Training Logs**

```bash
# Follow training progress
tail -f /workspace/nanochat-experiment/training.log

# Check GPU usage
watch -n 5 nvidia-smi

# Check specific metrics
grep "Validation loss" training.log
```

### **Key Metrics to Watch**

| Metric | Good | Warning | Action |
|--------|------|---------|--------|
| GPU Util | 95-100% | <80% | Check bottleneck |
| VRAM Usage | 75-78GB | >79GB | Reduce batch size |
| Loss | Decreasing | Increasing | Check overfitting |
| MFU | >20% | <15% | Check efficiency |
| Tok/sec | >50,000 | <30,000 | Check dataloader |

---

## ⚠️ Important Notes

### **Overfitting Prevention**

Given your dataset is 350M tokens and you're training for 31 hours:

```
Estimated steps: ~23,000
Tokens per step: 524,288
Total tokens: ~12B
Epochs: ~34 passes over dataset
```

**Risk:** High chance of overfitting (learned from $100 run)

**Solutions:**
1. **Early stopping**: Monitor validation loss, stop when it increases
2. **Use best checkpoint**: Don't use final checkpoint, use lowest val_loss
3. **More data**: Ideally 10x dataset size (3.5GB instead of 350MB)
4. **Regularization**: Add dropout or weight decay if available

### **Optimal Training Duration**

Based on $100 training results (converged at step 3000):

```
$100 training: Converged at 6% complete
$1000 training: May converge at 10-15% complete

Estimated optimal: 3-5 hours instead of 31 hours
Checkpoints to use: Steps 2000-5000
```

**Recommendation:** Plan to use early checkpoints (2-5 hours), not final checkpoint!

### **Disk Space Management**

```bash
# Check space during training
df -h /workspace

# If running low, delete old checkpoints:
rm /workspace/nanochat_experiment/checkpoints/d64/optim_*rank*.pt

# Keep only model weights (saves 50% space)
```

---

## 🔧 Troubleshooting

### **Out of Memory**

```bash
# Reduce batch size
torchrun --standalone --nproc_per_node=8 \
    -m experiment.train \
    --depth=64 \
    --device_batch_size=2  # ← Changed from 4
    --training_hours=31.0
```

### **Slow Training**

```bash
# Check GPU utilization
nvidia-smi dmon -s u

# If low, check:
# 1. Data loading (increase num_workers)
# 2. Gradient accumulation steps
# 3. CPU bottleneck
```

### **Connection Lost**

```bash
# Training continues in tmux!
# Just reconnect and reattach:
ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519
tmux attach -t train_d64
```

---

## 📊 Expected Results

### **Model Comparison**

| Model | Params | Val Loss | Use Case |
|-------|--------|----------|----------|
| **d24 (step_002920)** | 680M | 0.2627 | Production-ready |
| **d64 (step_~3000)** | 4.3B | 0.24-0.25 | Best quality (estimated) |
| **d64 (final)** | 4.3B | 0.20-0.22 | May overfit |

### **What to Expect**

**Good outcomes:**
- Lower validation loss than d24 (0.24-0.25 vs 0.26)
- Better generalization on test data
- Smoother, more coherent predictions

**Watch for:**
- Overfitting after 5-10 hours
- Training loss much lower than val loss
- Use checkpoints from first 5 hours

---

## 🎯 Post-Training

### **1. Identify Best Checkpoint**

```bash
# Find lowest validation loss
grep "Validation loss" training.log | sort -k5 -n | head -5
```

### **2. Download Inference-Only**

```bash
# Only model weights (no optimizer)
scp -P <PORT> -i ~/.ssh/id_ed25519 \
    root@<IP>:/workspace/nanochat_experiment/checkpoints/d64/model_XXXXXX.pt \
    root@<IP>:/workspace/nanochat_experiment/checkpoints/d64/meta_XXXXXX.json \
    ./d64_best/
```

### **3. Test Locally**

```bash
# Use same inference server
python3 -m experiment.server --checkpoint d64_best --port 8000
```

---

## 💡 Key Takeaways

1. ✅ **Depth 64 is optimal** for 80GB H100 GPUs
2. ✅ **Checkpoints saved to /workspace** (persistent)
3. ⚠️ **Watch for overfitting** around 5-10 hours
4. ⚠️ **Use early checkpoints**, not final
5. 💾 **Plan for ~527GB checkpoint storage**
6. 🎯 **Expected improvement**: 0.26 → 0.24-0.25 val loss

Good luck with your training! 🚀
