"""
Custom dataloader for the experiment.

This dataloader:
1. Loads text data from the experiment data directory
2. Tokenizes using our character-level tokenizer
3. Creates batches for training

Compatible with the nanochat training scripts.
"""

import os
import torch
from experiment.tokenizer import CharTokenizer, get_token_bytes
from experiment.dataset import load_all_data, get_experiment_base_dir


def get_tokenizer():
    """Get the character tokenizer."""
    return CharTokenizer()


def tokenizing_distributed_data_loader(batch_size, seq_len, split, device, rank=0, world_size=1):
    """
    A data loader that yields tokenized batches.
    
    Args:
        batch_size: Number of sequences per batch
        seq_len: Length of each sequence
        split: "train" or "val"
        device: torch device
        rank: DDP rank (for distributed training)
        world_size: DDP world size
    
    Yields:
        x: Input tokens (batch_size, seq_len)
        y: Target tokens (batch_size, seq_len)
    """
    # Load and tokenize all data
    data_dir = os.path.join(get_experiment_base_dir(), "data")
    text, _ = load_all_data(data_dir, validate=True)
    
    tokenizer = CharTokenizer()
    bos_token = tokenizer.get_bos_token_id()
    
    # Tokenize the entire dataset
    tokens = tokenizer.encode(text, prepend=bos_token)
    tokens = torch.tensor(tokens, dtype=torch.long)
    
    # Split into train/val (95%/5%)
    split_idx = int(len(tokens) * 0.95)
    if split == "train":
        tokens = tokens[:split_idx]
    else:
        tokens = tokens[split_idx:]
    
    print(f"[Rank {rank}] Loaded {len(tokens):,} tokens for {split} split")
    
    # Calculate number of complete sequences we can make
    n_tokens = len(tokens)
    tokens_per_batch = batch_size * (seq_len + 1) * world_size  # +1 for targets
    
    # For distributed: each rank gets a strided view
    # Shard the data across ranks
    rank_start = rank * batch_size * (seq_len + 1)
    
    # Infinite loop over the data
    idx = rank_start
    while True:
        # Collect batch_size sequences
        batch_x = []
        batch_y = []
        
        for _ in range(batch_size):
            # Wrap around if we've reached the end
            if idx + seq_len + 1 > n_tokens:
                idx = rank * batch_size * (seq_len + 1)  # Reset to start for this rank
            
            # Get a sequence
            seq = tokens[idx:idx + seq_len + 1]
            batch_x.append(seq[:-1])
            batch_y.append(seq[1:])
            
            # Advance by world_size to avoid overlap between ranks
            idx += (seq_len + 1) * world_size
            if idx >= n_tokens:
                idx = rank * batch_size * (seq_len + 1)
        
        x = torch.stack(batch_x).to(device)
        y = torch.stack(batch_y).to(device)
        
        yield x, y


def tokenizing_distributed_data_loader_with_state(batch_size, seq_len, split, device, resume_state_dict=None, rank=0, world_size=1):
    """
    Same as above but yields state dict for checkpointing.
    """
    # Load and tokenize all data
    data_dir = os.path.join(get_experiment_base_dir(), "data")
    text, _ = load_all_data(data_dir, validate=True)
    
    tokenizer = CharTokenizer()
    bos_token = tokenizer.get_bos_token_id()
    
    # Tokenize the entire dataset
    tokens = tokenizer.encode(text, prepend=bos_token)
    tokens = torch.tensor(tokens, dtype=torch.long)
    
    # Split into train/val (95%/5%)
    split_idx = int(len(tokens) * 0.95)
    if split == "train":
        tokens = tokens[:split_idx]
    else:
        tokens = tokens[split_idx:]
    
    n_tokens = len(tokens)
    print(f"[Rank {rank}] Loaded {len(tokens):,} tokens for {split} split")
    
    # Resume from state if provided
    if resume_state_dict is not None:
        idx = resume_state_dict.get("idx", 0)
        epoch = resume_state_dict.get("epoch", 0)
    else:
        idx = rank * batch_size * (seq_len + 1)
        epoch = 0
    
    while True:
        batch_x = []
        batch_y = []
        
        for _ in range(batch_size):
            if idx + seq_len + 1 > n_tokens:
                idx = rank * batch_size * (seq_len + 1)
                epoch += 1
                if rank == 0:
                    print(f"[Epoch {epoch}] Looping over data...")
            
            seq = tokens[idx:idx + seq_len + 1]
            batch_x.append(seq[:-1])
            batch_y.append(seq[1:])
            
            idx += (seq_len + 1) * world_size
            if idx >= n_tokens:
                idx = rank * batch_size * (seq_len + 1)
                epoch += 1
        
        x = torch.stack(batch_x).to(device)
        y = torch.stack(batch_y).to(device)
        
        state_dict = {"idx": idx, "epoch": epoch}
        yield x, y, state_dict
