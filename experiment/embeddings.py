"""
Embedding extraction utilities for trained models.

This module provides functions to extract embeddings (hidden states) from
a trained GPT model at various layers.

Usage:
    from experiment.embeddings import EmbeddingExtractor
    
    extractor = EmbeddingExtractor.from_checkpoint("d48")
    
    # Get final layer embeddings
    embeddings = extractor.get_embeddings("123+456")
    
    # Get embeddings from specific layer
    embeddings = extractor.get_embeddings("123+456", layer=-1)  # last layer
    embeddings = extractor.get_embeddings("123+456", layer=24)  # layer 24
    
    # Get all layer embeddings
    all_embeddings = extractor.get_all_layer_embeddings("123+456")
"""

import os
import sys
from contextlib import nullcontext

import torch
import torch.nn.functional as F

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanochat.gpt import GPT, GPTConfig, norm
from nanochat.checkpoint_manager import load_checkpoint
from nanochat.common import autodetect_device_type
from experiment.tokenizer import CharTokenizer, VOCAB_SIZE
from experiment.dataset import get_experiment_base_dir


class EmbeddingExtractor:
    """
    Extract embeddings from a trained GPT model.
    
    Embeddings can be extracted at different points:
    - After token embedding (layer 0)
    - After any transformer block (layers 1 to n_layer)
    - After final normalization (default, "final")
    """
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    @classmethod
    def from_checkpoint(cls, model_tag="d64", step=None, device=None):
        """
        Load a model from checkpoint and create an extractor.
        
        Args:
            model_tag: Model tag (e.g., "d48", "d64")
            step: Checkpoint step (None = latest)
            device: Device to load model on (None = auto)
        """
        if device is None:
            device_type = autodetect_device_type()
            device = torch.device(device_type)
        
        base_dir = get_experiment_base_dir()
        checkpoint_dir = os.path.join(base_dir, "checkpoints", model_tag)
        
        # Find latest step if not specified
        if step is None:
            # Look for checkpoint files
            import glob
            pattern = os.path.join(checkpoint_dir, "model_*.pt")
            checkpoints = glob.glob(pattern)
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
            # Extract step numbers and find max
            steps = []
            for cp in checkpoints:
                try:
                    s = int(os.path.basename(cp).replace("model_", "").replace(".pt", ""))
                    steps.append(s)
                except:
                    pass
            step = max(steps)
            print(f"Loading checkpoint from step {step}")
        
        # Load checkpoint
        model_data, _, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
        
        # Build model
        model_config_kwargs = meta_data["model_config"]
        model_config = GPTConfig(**model_config_kwargs)
        model = GPT(model_config, pad_vocab_size_to=32)
        model.to(device)
        model.load_state_dict(model_data, strict=True)
        model.eval()
        
        tokenizer = CharTokenizer()
        
        return cls(model, tokenizer, device)
    
    @torch.inference_mode()
    def get_embeddings(self, text, layer="final", pooling="last"):
        """
        Get embeddings for input text.
        
        Args:
            text: Input string or list of token IDs
            layer: Which layer to extract from:
                   - "final": After all blocks + final norm (default)
                   - "token": Just token embeddings (before any blocks)
                   - int: After specific block (0 to n_layer-1)
                   - -1: Same as "final"
            pooling: How to pool sequence embeddings:
                     - "last": Take last token's embedding (default)
                     - "mean": Average all token embeddings
                     - "first": Take first token's embedding
                     - "none": Return all token embeddings (B, T, D)
        
        Returns:
            torch.Tensor: Embeddings of shape (D,) or (B, D) or (B, T, D)
        """
        # Tokenize if needed
        if isinstance(text, str):
            tokens = self.tokenizer.encode(text, prepend="<|bos|>")
        else:
            tokens = text
        
        # Convert to tensor
        idx = torch.tensor([tokens], dtype=torch.long, device=self.device)
        B, T = idx.size()
        
        # Get rotary embeddings
        T0 = 0
        cos_sin = self.model.cos[:, T0:T0+T], self.model.sin[:, T0:T0+T]
        
        # Forward through model, stopping at requested layer
        x = self.model.transformer.wte(idx)
        x = norm(x)
        
        if layer == "token":
            embeddings = x
        else:
            n_layers = len(self.model.transformer.h)
            
            # Determine which layer to stop at
            if layer == "final" or layer == -1:
                stop_at = n_layers
            elif isinstance(layer, int):
                stop_at = min(layer + 1, n_layers)
            else:
                stop_at = n_layers
            
            # Forward through blocks
            for i, block in enumerate(self.model.transformer.h):
                if i >= stop_at:
                    break
                x = block(x, cos_sin, kv_cache=None)
            
            # Apply final norm if going to the end
            if layer == "final" or layer == -1 or layer >= n_layers - 1:
                x = norm(x)
            
            embeddings = x
        
        # Apply pooling
        if pooling == "last":
            embeddings = embeddings[:, -1, :]  # (B, D)
        elif pooling == "mean":
            embeddings = embeddings.mean(dim=1)  # (B, D)
        elif pooling == "first":
            embeddings = embeddings[:, 0, :]  # (B, D)
        elif pooling == "none":
            pass  # Keep (B, T, D)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
        
        # Squeeze batch dim if single input
        if embeddings.size(0) == 1 and pooling != "none":
            embeddings = embeddings.squeeze(0)  # (D,)
        
        return embeddings
    
    @torch.inference_mode()
    def get_all_layer_embeddings(self, text, pooling="last"):
        """
        Get embeddings from ALL layers.
        
        Returns:
            dict: {layer_name: embedding_tensor}
                  Keys: "token", "block_0", ..., "block_N-1", "final"
        """
        # Tokenize
        if isinstance(text, str):
            tokens = self.tokenizer.encode(text, prepend="<|bos|>")
        else:
            tokens = text
        
        idx = torch.tensor([tokens], dtype=torch.long, device=self.device)
        B, T = idx.size()
        
        cos_sin = self.model.cos[:, :T], self.model.sin[:, :T]
        
        embeddings = {}
        
        # Token embeddings
        x = self.model.transformer.wte(idx)
        x = norm(x)
        embeddings["token"] = self._pool(x, pooling)
        
        # Each block
        for i, block in enumerate(self.model.transformer.h):
            x = block(x, cos_sin, kv_cache=None)
            embeddings[f"block_{i}"] = self._pool(x, pooling)
        
        # Final norm
        x = norm(x)
        embeddings["final"] = self._pool(x, pooling)
        
        return embeddings
    
    def _pool(self, x, pooling):
        """Apply pooling to embeddings."""
        if pooling == "last":
            return x[:, -1, :].squeeze(0)
        elif pooling == "mean":
            return x.mean(dim=1).squeeze(0)
        elif pooling == "first":
            return x[:, 0, :].squeeze(0)
        else:
            return x.squeeze(0)
    
    @torch.inference_mode()
    def get_similarity(self, text1, text2, layer="final"):
        """
        Compute cosine similarity between two texts.
        """
        emb1 = self.get_embeddings(text1, layer=layer, pooling="last")
        emb2 = self.get_embeddings(text2, layer=layer, pooling="last")
        
        similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
        return similarity.item()
    
    @torch.inference_mode()
    def batch_embeddings(self, texts, layer="final", pooling="last", max_len=None):
        """
        Get embeddings for a batch of texts.
        
        Args:
            texts: List of strings
            layer: Which layer
            pooling: How to pool
            max_len: Max sequence length (for padding)
        
        Returns:
            torch.Tensor: (batch_size, embedding_dim)
        """
        # Tokenize all
        all_tokens = [self.tokenizer.encode(t, prepend="<|bos|>") for t in texts]
        
        # Find max length
        if max_len is None:
            max_len = max(len(t) for t in all_tokens)
        
        # Pad sequences
        batch = []
        for tokens in all_tokens:
            if len(tokens) < max_len:
                # Pad with zeros (will be masked)
                tokens = tokens + [0] * (max_len - len(tokens))
            batch.append(tokens[:max_len])
        
        # Get embeddings for each (could optimize with true batching)
        embeddings = []
        for tokens in batch:
            emb = self.get_embeddings(tokens, layer=layer, pooling=pooling)
            embeddings.append(emb)
        
        return torch.stack(embeddings)


def main():
    """Demo usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract embeddings from trained model")
    parser.add_argument("--model", type=str, default="d24", help="Model tag (d24 for $100 budget, d48 for $1000 budget)")
    parser.add_argument("--text", type=str, default="123+456", help="Input text")
    parser.add_argument("--layer", type=str, default="final", help="Layer to extract from")
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    try:
        extractor = EmbeddingExtractor.from_checkpoint(args.model)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure you've trained a model first with: bash experiment/run.sh")
        return
    
    print(f"\nExtracting embeddings for: '{args.text}'")
    print(f"Layer: {args.layer}")
    
    # Parse layer arg
    layer = args.layer
    if layer.isdigit():
        layer = int(layer)
    elif layer == "-1":
        layer = -1
    
    embeddings = extractor.get_embeddings(args.text, layer=layer)
    
    print(f"\nEmbedding shape: {embeddings.shape}")
    print(f"Embedding dtype: {embeddings.dtype}")
    print(f"Embedding norm: {embeddings.norm().item():.4f}")
    print(f"First 10 values: {embeddings[:10].tolist()}")
    
    # Show similarity example
    print("\n--- Similarity Example ---")
    texts = ["123+456", "123+457", "999-000", "HB+123"]
    print(f"Comparing against: '{texts[0]}'")
    for t in texts[1:]:
        sim = extractor.get_similarity(texts[0], t)
        print(f"  '{t}': {sim:.4f}")


if __name__ == "__main__":
    main()
