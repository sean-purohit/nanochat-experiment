"""
Custom character-level tokenizer for the experiment.

Vocabulary:
- Digits: 0-9 (10 tokens)
- Operators: +, - (2 tokens)
- Letters: H, B, L (3 tokens)
- Decimal: . (1 token)
- Newline: \n (1 token)
- Special token: <|bos|> (1 token)

Total: 18 tokens (padded to 32 for efficiency)
"""

import os
import json

# The allowed characters in our vocabulary
VOCAB_CHARS = list("0123456789+-HBL.\n")

# Special tokens - minimal, no chat interface needed
SPECIAL_TOKENS = [
    "<|bos|>",
    # Chat tokens removed - not using chat interface
    # "<|user_start|>",
    # "<|user_end|>",
    # "<|assistant_start|>",
    # "<|assistant_end|>",
    # "<|python_start|>",
    # "<|python_end|>",
    # "<|output_start|>",
    # "<|output_end|>",
]

# Build the vocabulary
# Characters get IDs 0-16, special tokens get 17+, padding to 32
CHAR_TO_ID = {char: i for i, char in enumerate(VOCAB_CHARS)}
ID_TO_CHAR = {i: char for char, i in CHAR_TO_ID.items()}

SPECIAL_TOKEN_OFFSET = len(VOCAB_CHARS)
SPECIAL_TO_ID = {token: SPECIAL_TOKEN_OFFSET + i for i, token in enumerate(SPECIAL_TOKENS)}
ID_TO_SPECIAL = {i: token for token, i in SPECIAL_TO_ID.items()}

# Total vocab size (padded to 32 for nice divisibility)
VOCAB_SIZE = 32


class CharTokenizer:
    """
    Simple character-level tokenizer for a restricted vocabulary.
    Compatible with the nanochat tokenizer interface.
    """
    
    def __init__(self):
        self.char_to_id = CHAR_TO_ID.copy()
        self.id_to_char = ID_TO_CHAR.copy()
        self.special_to_id = SPECIAL_TO_ID.copy()
        self.id_to_special = ID_TO_SPECIAL.copy()
        self.vocab_size = VOCAB_SIZE
        self.bos_token_id = self.special_to_id["<|bos|>"]
    
    @classmethod
    def from_directory(cls, tokenizer_dir):
        """Load tokenizer from directory (for compatibility, we don't actually need to load anything)."""
        return cls()
    
    def get_vocab_size(self):
        return self.vocab_size
    
    def get_special_tokens(self):
        return list(self.special_to_id.keys())
    
    def get_bos_token_id(self):
        return self.bos_token_id
    
    def id_to_token(self, id):
        if id in self.id_to_char:
            return self.id_to_char[id]
        elif id in self.id_to_special:
            return self.id_to_special[id]
        else:
            return f"<|unk_{id}|>"
    
    def encode_special(self, text):
        """Encode a single special token."""
        return self.special_to_id.get(text)
    
    def _encode_one(self, text, prepend=None, append=None):
        """Encode a single string to token IDs."""
        ids = []
        
        # Handle prepend
        if prepend is not None:
            if isinstance(prepend, int):
                ids.append(prepend)
            else:
                ids.append(self.encode_special(prepend))
        
        # Encode the text character by character
        for char in text:
            if char in self.char_to_id:
                ids.append(self.char_to_id[char])
            # Skip unknown characters (or you could raise an error)
        
        # Handle append
        if append is not None:
            if isinstance(append, int):
                ids.append(append)
            else:
                ids.append(self.encode_special(append))
        
        return ids
    
    def encode(self, text, prepend=None, append=None, num_threads=8):
        """Encode text or list of texts to token IDs."""
        if isinstance(text, str):
            return self._encode_one(text, prepend, append)
        elif isinstance(text, list):
            return [self._encode_one(t, prepend, append) for t in text]
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
    
    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)
    
    def decode(self, ids):
        """Decode token IDs back to text."""
        result = []
        for id in ids:
            if id in self.id_to_char:
                result.append(self.id_to_char[id])
            elif id in self.id_to_special:
                result.append(self.id_to_special[id])
            # Skip padding tokens (IDs 24-31)
        return "".join(result)
    
    def save(self, tokenizer_dir):
        """Save tokenizer config to directory."""
        os.makedirs(tokenizer_dir, exist_ok=True)
        config = {
            "vocab_chars": VOCAB_CHARS,
            "special_tokens": SPECIAL_TOKENS,
            "vocab_size": self.vocab_size,
        }
        config_path = os.path.join(tokenizer_dir, "tokenizer_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Saved tokenizer config to {config_path}")
    
    # Chat interface methods removed - not using chat format for this experiment
    # See original nanochat tokenizer if chat support is needed


def get_tokenizer():
    """Get the tokenizer instance (for compatibility with nanochat imports)."""
    return CharTokenizer()


def get_token_bytes(device="cpu"):
    """
    Get token bytes tensor for BPB calculation.
    For character-level tokenizer, each token is 1 byte (for the base chars).
    """
    import torch
    
    # Create a tensor where each entry is the byte length of that token
    token_bytes = torch.ones(VOCAB_SIZE, dtype=torch.long, device=device)
    
    # Special tokens have their string length as bytes
    for token, id in SPECIAL_TO_ID.items():
        token_bytes[id] = len(token.encode('utf-8'))
    
    # Padding tokens (24-31) are 0 bytes
    for i in range(len(VOCAB_CHARS) + len(SPECIAL_TOKENS), VOCAB_SIZE):
        token_bytes[i] = 0
    
    return token_bytes


if __name__ == "__main__":
    # Quick test
    tokenizer = CharTokenizer()
    print(f"Vocab size: {tokenizer.get_vocab_size()}")
    print(f"Special tokens: {tokenizer.get_special_tokens()}")
    print(f"BOS token ID: {tokenizer.get_bos_token_id()}")
    
    # Test encoding/decoding
    test_text = "123+456-789\nHB"
    encoded = tokenizer.encode(test_text, prepend="<|bos|>")
    decoded = tokenizer.decode(encoded)
    print(f"Original: {repr(test_text)}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {repr(decoded)}")
