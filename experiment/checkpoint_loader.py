"""
Standalone checkpoint loader for experiment models.
Avoids dependencies on nanochat.checkpoint_manager which requires rustbpe.
"""

import os
import json
import torch


def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0):
    """
    Load a checkpoint from disk.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        step: Training step to load
        device: Device to load tensors to
        load_optimizer: Whether to load optimizer state
        rank: GPU rank (for multi-GPU training)
    
    Returns:
        (model_state_dict, optimizer_state_dict, metadata)
    """
    # Load model
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    model_data = torch.load(model_path, map_location=device)
    
    # Load metadata
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    
    with open(meta_path, 'r') as f:
        meta_data = json.load(f)
    
    # Load optimizer if requested
    optimizer_data = None
    if load_optimizer:
        optim_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank}.pt")
        if os.path.exists(optim_path):
            optimizer_data = torch.load(optim_path, map_location=device)
    
    return model_data, optimizer_data, meta_data
