#!/usr/bin/env python3
"""
Example client for the Inference API.

This demonstrates how to call the API from Python code.

Usage:
    python experiment/client_example.py
"""

import requests
import json


class InferenceClient:
    """Client for the Inference API."""
    
    def __init__(self, base_url="http://127.0.0.1:8000"):
        self.base_url = base_url
    
    def health(self):
        """Check server health."""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def info(self):
        """Get model information."""
        response = requests.get(f"{self.base_url}/info")
        response.raise_for_status()
        return response.json()
    
    def generate(self, prompt, max_tokens=100, temperature=0.8, top_k=None, stop_on_newline=False):
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            top_k: Top-k sampling (None = use all tokens)
            stop_on_newline: Stop at newline
        
        Returns:
            Dictionary with 'prompt', 'generated_text', 'tokens_generated', 'model'
        """
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop_on_newline": stop_on_newline
        }
        
        if top_k is not None:
            payload["top_k"] = top_k
        
        response = requests.post(f"{self.base_url}/generate", json=payload)
        response.raise_for_status()
        return response.json()


def main():
    """Example usage."""
    
    # Create client
    client = InferenceClient("http://127.0.0.1:8000")
    
    print("="*60)
    print("Inference API Client Example")
    print("="*60)
    print()
    
    # Health check
    print("1. Health Check:")
    health = client.health()
    print(f"   Status: {health['status']}")
    print(f"   Model loaded: {health['model_loaded']}")
    print(f"   Checkpoint: {health['checkpoint']}")
    print()
    
    # Model info
    print("2. Model Information:")
    info = client.info()
    print(f"   Checkpoint: {info['checkpoint']}")
    print(f"   Step: {info['step']}")
    print(f"   Val Loss: {info['val_loss']:.4f}")
    print(f"   Layers: {info['model_layers']}")
    print(f"   Model Dim: {info['model_dim']}")
    print(f"   Vocab Size: {info['vocab_size']}")
    print()
    
    # Generate text - Example 1
    print("3. Generation Example 1 (Greedy):")
    print("   Prompt: '123+456='")
    result = client.generate(
        prompt="123+456=",
        max_tokens=20,
        temperature=0.0  # Greedy decoding
    )
    print(f"   Generated: {result['generated_text']}")
    print(f"   Tokens: {result['tokens_generated']}")
    print()
    
    # Generate text - Example 2
    print("4. Generation Example 2 (Sampling):")
    print("   Prompt: 'H+'")
    result = client.generate(
        prompt="H+",
        max_tokens=50,
        temperature=0.8,
        top_k=10
    )
    print(f"   Generated: {result['generated_text']}")
    print(f"   Tokens: {result['tokens_generated']}")
    print()
    
    # Generate text - Example 3
    print("5. Generation Example 3 (Stop on newline):")
    print("   Prompt: '0+'")
    result = client.generate(
        prompt="0+",
        max_tokens=100,
        temperature=0.5,
        stop_on_newline=True
    )
    print(f"   Generated: {result['generated_text']}")
    print(f"   Tokens: {result['tokens_generated']}")
    print()
    
    print("="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to server.")
        print("Make sure the server is running:")
        print("  python -m experiment.server --checkpoint step_002920")
    except Exception as e:
        print(f"Error: {e}")
