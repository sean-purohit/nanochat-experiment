#!/usr/bin/env python3
"""
Inference script for trained models.

Usage:
    # Generate text from a prompt
    python -m experiment.inference --checkpoint step_002920 --prompt "123+456="
    
    # Interactive mode
    python -m experiment.inference --checkpoint step_002920 --interactive
    
    # Specify number of tokens to generate
    python -m experiment.inference --checkpoint step_002920 --prompt "H+" --max_tokens 50
"""

import os
import sys
import argparse
from contextlib import nullcontext

import torch
import torch.nn.functional as F

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanochat.gpt import GPT, GPTConfig
from nanochat.common import autodetect_device_type
from experiment.tokenizer import CharTokenizer, VOCAB_SIZE
from experiment.dataset import get_experiment_base_dir
from experiment.checkpoint_loader import load_checkpoint


class InferenceEngine:
    """
    Simple inference engine for text generation.
    """
    
    def __init__(self, model, tokenizer, device, temperature=0.8, top_k=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.temperature = temperature
        self.top_k = top_k
        self.model.eval()
    
    @classmethod
    def from_checkpoint(cls, checkpoint_name, step=None, device=None, temperature=0.8, top_k=None):
        """
        Load a model from checkpoint directory.
        
        Args:
            checkpoint_name: Name of checkpoint folder (e.g., "step_002920")
            step: Checkpoint step (None = auto-detect from folder)
            device: Device to load model on (None = auto)
            temperature: Sampling temperature (0.0 = greedy, 1.0 = random)
            top_k: Top-k sampling (None = use all tokens)
        """
        if device is None:
            device_type = autodetect_device_type()
            device = torch.device(device_type)
        
        # Get checkpoint directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_dir = os.path.join(script_dir, "checkpoints", checkpoint_name)
        
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        
        # Auto-detect step from folder name if not provided
        if step is None:
            # Try to extract from folder name (e.g., "step_002920" -> 2920)
            if checkpoint_name.startswith("step_"):
                try:
                    step = int(checkpoint_name.replace("step_", ""))
                except:
                    pass
            
            # If still None, look for model files
            if step is None:
                import glob
                pattern = os.path.join(checkpoint_dir, "model_*.pt")
                checkpoints = glob.glob(pattern)
                if not checkpoints:
                    raise FileNotFoundError(f"No model files found in {checkpoint_dir}")
                # Extract step from filename
                step = int(os.path.basename(checkpoints[0]).replace("model_", "").replace(".pt", ""))
        
        print(f"Loading checkpoint from {checkpoint_name} at step {step}")
        
        # Load checkpoint
        model_data, _, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
        
        # Build model
        model_config_kwargs = meta_data["model_config"]
        model_config = GPTConfig(**model_config_kwargs)
        model = GPT(model_config, pad_vocab_size_to=32)
        model.to(device)
        model.load_state_dict(model_data, strict=True)
        
        print(f"Model loaded: {model_config.n_layer} layers, {model_config.n_embd} dim")
        print(f"Validation loss at this checkpoint: {meta_data.get('val_loss', 'N/A'):.4f}")
        
        # Create tokenizer
        tokenizer = CharTokenizer()
        
        return cls(model, tokenizer, device, temperature, top_k)
    
    @torch.no_grad()
    def generate(self, prompt, max_tokens=100, temperature=None, top_k=None, stop_on_newline=False):
        """
        Generate text continuation from a prompt.
        
        Args:
            prompt: Input text string
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (None = use default)
            top_k: Top-k sampling (None = use default)
            stop_on_newline: Stop generation at newline
        
        Returns:
            Generated text (including prompt)
        """
        temperature = temperature if temperature is not None else self.temperature
        top_k = top_k if top_k is not None else self.top_k
        
        # Encode prompt
        tokens = self.tokenizer.encode(prompt, prepend="<|bos|>")
        tokens = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        
        # Generate tokens
        for _ in range(max_tokens):
            # Forward pass
            with torch.no_grad():
                logits = self.model(tokens)
            
            # Get logits for last token
            logits = logits[:, -1, :].clone()
            
            # Apply temperature
            if temperature > 0:
                logits = logits / temperature
            
            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            
            # Handle edge case of all -inf (shouldn't happen but safety)
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                # Fallback to uniform distribution over valid tokens
                probs = torch.ones_like(probs) / probs.size(-1)
            
            if temperature == 0:
                # Greedy decoding
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                # Sample from distribution
                next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            tokens = torch.cat([tokens, next_token], dim=1)
            
            # Check for stopping conditions
            next_token_id = next_token.item()
            if stop_on_newline:
                next_char = self.tokenizer.id_to_token(next_token_id)
                if next_char == '\n':
                    break
            
            # Stop if sequence is too long
            if tokens.size(1) >= 2048:  # Max sequence length
                break
        
        # Decode
        generated_tokens = tokens[0].tolist()
        # Skip BOS token in output
        if generated_tokens[0] == self.tokenizer.bos_token_id:
            generated_tokens = generated_tokens[1:]
        
        return self.tokenizer.decode(generated_tokens)
    
    def interactive(self):
        """
        Interactive generation mode.
        """
        print("\n" + "="*60)
        print("Interactive Inference Mode")
        print("="*60)
        print(f"Temperature: {self.temperature}")
        print(f"Top-k: {self.top_k if self.top_k else 'None (use all tokens)'}")
        print("\nEnter prompts (empty line to quit)")
        print("Commands:")
        print("  /temp <value>  - Set temperature (e.g., /temp 0.5)")
        print("  /topk <value>  - Set top-k (e.g., /topk 10)")
        print("  /tokens <n>    - Set max tokens (e.g., /tokens 50)")
        print("  /quit          - Exit")
        print("="*60 + "\n")
        
        max_tokens = 100
        
        while True:
            try:
                prompt = input("Prompt: ").strip()
                
                if not prompt:
                    break
                
                # Handle commands
                if prompt.startswith("/"):
                    parts = prompt.split()
                    cmd = parts[0].lower()
                    
                    if cmd == "/quit":
                        break
                    elif cmd == "/temp" and len(parts) == 2:
                        self.temperature = float(parts[1])
                        print(f"Temperature set to {self.temperature}")
                        continue
                    elif cmd == "/topk" and len(parts) == 2:
                        self.top_k = int(parts[1]) if parts[1] != "none" else None
                        print(f"Top-k set to {self.top_k}")
                        continue
                    elif cmd == "/tokens" and len(parts) == 2:
                        max_tokens = int(parts[1])
                        print(f"Max tokens set to {max_tokens}")
                        continue
                    else:
                        print("Unknown command")
                        continue
                
                # Generate
                output = self.generate(prompt, max_tokens=max_tokens)
                print(f"\nGenerated:\n{output}\n")
                print("-" * 60)
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run inference with trained model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Checkpoint folder name (e.g., step_002920)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Prompt for generation")
    parser.add_argument("--max_tokens", type=int, default=100,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (0.0=greedy, 1.0=random)")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Top-k sampling (None=use all tokens)")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cpu/cuda/mps, default=auto)")
    
    args = parser.parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device_type = autodetect_device_type()
        device = torch.device(device_type)
    
    print(f"Using device: {device}")
    
    # Load model
    engine = InferenceEngine.from_checkpoint(
        args.checkpoint,
        device=device,
        temperature=args.temperature,
        top_k=args.top_k
    )
    
    # Run inference
    if args.interactive:
        engine.interactive()
    elif args.prompt:
        output = engine.generate(args.prompt, max_tokens=args.max_tokens)
        print(f"\nPrompt: {args.prompt}")
        print(f"Generated:\n{output}\n")
    else:
        print("Error: Provide either --prompt or --interactive")
        sys.exit(1)


if __name__ == "__main__":
    main()
