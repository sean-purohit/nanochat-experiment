"""
Training script for the experiment.

This is a modified version of nanochat's base_train.py adapted for:
- Custom character-level tokenizer (15 base chars + 9 special = 24 tokens, padded to 32)
- Custom dataset loader
- Deeper models (d48-d64) due to tiny vocabulary
- Many epochs over small dataset

Run as:
    python -m experiment.train

Or distributed:
    torchrun --standalone --nproc_per_node=8 -m experiment.train
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import sys
import argparse
import time
from contextlib import nullcontext

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, autodetect_device_type
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.engine import Engine

# Import our custom modules
from experiment.tokenizer import CharTokenizer, get_token_bytes, VOCAB_SIZE
from experiment.dataloader import tokenizing_distributed_data_loader, tokenizing_distributed_data_loader_with_state
from experiment.dataset import get_experiment_base_dir

print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Train model on custom dataset")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device_type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")

# =============================================================================
# $100 BUDGET CONFIGURATION (CURRENT) - 2 GPUs
# =============================================================================
# Model architecture - moderate depth for $100 budget
parser.add_argument("--depth", type=int, default=24, help="depth of the Transformer model (default: 24 for $100 budget)")
parser.add_argument("--aspect_ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head_dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max_seq_len", type=int, default=2048, help="max context length")
# Training horizon
parser.add_argument("--num_iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = auto)")
parser.add_argument("--training_hours", type=float, default=16.0, help="target training hours (2 GPUs × 16h × $3 = $96)")
# Optimization
parser.add_argument("--device_batch_size", type=int, default=8, help="per-device batch size (larger for smaller model)")
parser.add_argument("--total_batch_size", type=int, default=524288, help="total batch size in tokens")

# =============================================================================
# $1000 BUDGET CONFIGURATION (UNCOMMENT FOR FULL TRAINING) - 8 GPUs
# =============================================================================
# # Model architecture - DEEPER than standard nanochat due to tiny vocab
# parser.add_argument("--depth", type=int, default=48, help="depth of the Transformer model (default: 48, deeper than standard)")
# parser.add_argument("--aspect_ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
# parser.add_argument("--head_dim", type=int, default=128, help="target head dimension for attention")
# parser.add_argument("--max_seq_len", type=int, default=2048, help="max context length")
# # Training horizon
# parser.add_argument("--num_iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = auto)")
# parser.add_argument("--training_hours", type=float, default=31.0, help="target training hours (8 GPUs × 31h × $3 = $744)")
# # Optimization
# parser.add_argument("--device_batch_size", type=int, default=4, help="per-device batch size (smaller due to deeper model)")
parser.add_argument("--embedding_lr", type=float, default=0.3, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding_lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
parser.add_argument("--matrix_lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--adam_beta1", type=float, default=0.8, help="Adam beta1")
parser.add_argument("--adam_beta2", type=float, default=0.95, help="Adam beta2")
parser.add_argument("--warmup_ratio", type=float, default=0.0, help="ratio of iterations for LR warmup")
parser.add_argument("--warmdown_ratio", type=float, default=0.4, help="ratio of iterations for LR warmdown")
parser.add_argument("--final_lr_frac", type=float, default=0.0, help="final LR as fraction of initial LR")
parser.add_argument("--resume_from_step", type=int, default=-1, help="resume training from this step")
# Evaluation
parser.add_argument("--eval_every", type=int, default=500, help="evaluate val loss every N steps (-1 = disable)")
parser.add_argument("--sample_every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save_every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
# Output
parser.add_argument("--model_tag", type=str, default=None, help="override model tag for checkpoint directory")
args = parser.parse_args()
user_config = vars(args).copy()

# -----------------------------------------------------------------------------
# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# wandb logging
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-experiment", name=args.run, config=user_config)

# Tokenizer
tokenizer = CharTokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = VOCAB_SIZE  # 32 (padded)
print0(f"Vocab size: {vocab_size:,} (15 chars + 9 special + 8 padding)")

# Model architecture derived from depth
num_layers = args.depth
model_dim = args.depth * args.aspect_ratio

def find_num_heads(model_dim, target_head_dim):
    ideal = max(1, round(model_dim / target_head_dim))
    for offset in range(model_dim):
        for candidate in [ideal + offset, ideal - offset]:
            if candidate > 0 and model_dim % candidate == 0:
                return candidate
    return 1

num_heads = find_num_heads(model_dim, args.head_dim)
num_kv_heads = num_heads

print0(f"num_layers: {num_layers}")
print0(f"model_dim: {model_dim}")
print0(f"num_heads: {num_heads}")
print0(f"num_kv_heads: {num_kv_heads}")

# Batch size calculations
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert args.total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {args.total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# LR scaling
batch_lr_scale = 1.0
reference_batch_size = 2**19
batch_ratio = args.total_batch_size / reference_batch_size
if batch_ratio != 1.0:
    batch_lr_scale = batch_ratio ** 0.5
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {args.total_batch_size:,}")

# -----------------------------------------------------------------------------
# Initialize Model
model_config_kwargs = dict(
    sequence_len=args.max_seq_len,
    vocab_size=vocab_size,
    n_layer=num_layers,
    n_head=num_heads,
    n_kv_head=num_kv_heads,
    n_embd=model_dim
)

with torch.device("meta"):
    model_config = GPTConfig(**model_config_kwargs)
    model = GPT(model_config, pad_vocab_size_to=32)  # Pad to 32 for our vocab

model.to_empty(device=device)
model.init_weights()

# Checkpoint directory
base_dir = get_experiment_base_dir()
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}"
checkpoint_dir = os.path.join(base_dir, "checkpoints", output_dirname)

resuming = args.resume_from_step != -1
if resuming:
    print0(f"Resuming from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data

orig_model = model
model = torch.compile(model, dynamic=False)

num_params = sum(p.numel() for p in model.parameters())
num_scaling_params = orig_model.num_scaling_params()
print0(f"Number of parameters: {num_params:,}")
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# Calculate number of iterations
# Note: With tiny vocab, embeddings are negligible, so params ≈ transformer layers only
# $100 budget: d24 at 1536 dim ≈ 24 * 25M = ~600M params, ~9K iterations fills 4 hours
# $1000 budget: d48 at 3072 dim ≈ 48 * 50.4M = 2.42B params, ~47K iterations fills 31 hours

if args.num_iterations > 0:
    num_iterations = args.num_iterations
    print0(f"Using user-provided iterations: {num_iterations:,}")
else:
    # Estimate iterations based on target training time
    # d48 roughly: 2.3s per iteration on 8xH100
    # Estimate step time based on depth (d32 = 1.57s, scale roughly linearly)
    estimated_step_time = 1.57 * (args.depth / 32)
    num_iterations = int(args.training_hours * 3600 / estimated_step_time)
    print0(f"Estimated ~{estimated_step_time:.2f}s per step for d{args.depth}")
    print0(f"Calculated {num_iterations:,} iterations for {args.training_hours}h training")

total_tokens = args.total_batch_size * num_iterations
print0(f"Total training tokens: {total_tokens:,}")
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# With 350M tokens dataset:
estimated_dataset_tokens = 350_000_000  # 350M
epochs = total_tokens / estimated_dataset_tokens
print0(f"Estimated epochs over 350M token dataset: {epochs:.1f}")

# -----------------------------------------------------------------------------
# Initialize Optimizers
adam_betas = (args.adam_beta1, args.adam_beta2)
optimizers = model.setup_optimizers(
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=args.weight_decay,
    adam_betas=adam_betas,
)
adamw_optimizer, muon_optimizer = optimizers

if resuming:
    for opt, dat in zip(optimizers, optimizer_data):
        opt.load_state_dict(dat)
    del optimizer_data

# -----------------------------------------------------------------------------
# Initialize DataLoaders
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
train_loader = tokenizing_distributed_data_loader_with_state(
    args.device_batch_size, args.max_seq_len, split="train", device=device,
    resume_state_dict=dataloader_resume_state_dict, rank=ddp_rank, world_size=ddp_world_size
)
build_val_loader = lambda: tokenizing_distributed_data_loader(
    args.device_batch_size, args.max_seq_len, split="val", device=device,
    rank=ddp_rank, world_size=ddp_world_size
)
x, y, dataloader_state_dict = next(train_loader)

# -----------------------------------------------------------------------------
# LR Scheduler
def get_lr_multiplier(it):
    warmup_iters = round(args.warmup_ratio * num_iterations)
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac

def get_muon_momentum(it):
    frac = min(it / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

# -----------------------------------------------------------------------------
# Training Loop State
if not resuming:
    step = 0
    val_loss = None
    min_val_loss = float("inf")
    smooth_train_loss = 0
    total_training_time = 0
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_loss = meta_data.get("val_loss")
    min_val_loss = loop_state["min_val_loss"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# -----------------------------------------------------------------------------
# Training Loop
print0("\n" + "="*60)
print0("Starting training...")
print0("="*60 + "\n")

while True:
    last_step = step == num_iterations
    
    # Evaluation
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        val_losses = []
        eval_steps = 20  # Quick eval
        for i, (vx, vy) in enumerate(val_loader):
            if i >= eval_steps:
                break
            with autocast_ctx:
                loss = orig_model(vx, vy)
            val_losses.append(loss.item())
        val_loss = sum(val_losses) / len(val_losses) if val_losses else 0
        print0(f"Step {step:05d} | Validation loss: {val_loss:.4f}")
        if val_loss < min_val_loss:
            min_val_loss = val_loss
        wandb_run.log({"step": step, "val/loss": val_loss})
        model.train()
    
    # Sampling
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompts = [
            "123+456",
            "999-111",
            "HB+123",
            "0+0",
        ]
        engine = Engine(orig_model, tokenizer)
        print0("\n--- Samples ---")
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with autocast_ctx:
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=32, temperature=0.8)
            print0(f"  {prompt} -> {tokenizer.decode(sample[0])}")
        print0("---------------\n")
        model.train()
    
    # Checkpointing
    if last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
        save_checkpoint(
            checkpoint_dir, step,
            orig_model.state_dict(),
            [opt.state_dict() for opt in optimizers],
            {
                "step": step,
                "val_loss": val_loss,
                "model_config": model_config_kwargs,
                "user_config": user_config,
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": {
                    "min_val_loss": min_val_loss,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )
    
    if last_step:
        break
    
    # Training step
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, dataloader_state_dict = next(train_loader)
    
    # Optimizer step
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    muon_momentum = get_muon_momentum(step)
    for group in muon_optimizer.param_groups:
        group["momentum"] = muon_momentum
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    
    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(args.total_batch_size / dt)
    flops_per_sec = num_flops_per_token * args.total_batch_size / dt
    promised_flops = 989e12 * ddp_world_size
    mfu = 100 * flops_per_sec / promised_flops
    
    if step > 10:
        total_training_time += dt
    
    steps_done = step - 10
    if steps_done > 0:
        avg_time = total_training_time / steps_done
        remaining = num_iterations - step
        eta = remaining * avg_time
        eta_str = f" | eta: {eta/3600:.1f}h"
    else:
        eta_str = ""
    
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f}{eta_str}")
    
    if step % 100 == 0:
        wandb_run.log({
            "step": step,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
        })
    
    step += 1

# Final stats
print0(f"\n{'='*60}")
print0(f"Training complete!")
print0(f"Peak memory: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/3600:.2f}h")
if val_loss is not None:
    print0(f"Min validation loss: {min_val_loss:.4f}")
print0(f"{'='*60}\n")

wandb_run.finish()
compute_cleanup()
