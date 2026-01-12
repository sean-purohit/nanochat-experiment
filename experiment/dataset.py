"""
Custom dataset loader for the experiment.

Expects data to be placed in: ~/.cache/nanochat_experiment/data/
Data format: Plain text files (.txt) containing only valid characters:
- Digits: 0-9
- Operators: +, -
- Letters: H, B
- Newlines: \n

The loader will:
1. Scan the data directory for all .txt files
2. Read and concatenate all text
3. Validate that only allowed characters are present
4. Yield batches of documents for training
"""

import os
import glob
from pathlib import Path

# Allowed characters in our vocabulary
ALLOWED_CHARS = set("0123456789+-HBL.\n")


def get_experiment_base_dir():
    """Get the base directory for experiment artifacts."""
    return os.path.join(os.path.expanduser("~"), ".cache", "nanochat_experiment")


def get_data_dir():
    """Get the data directory."""
    base_dir = get_experiment_base_dir()
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def validate_text(text, filename="<unknown>"):
    """Validate that text only contains allowed characters."""
    invalid_chars = set()
    for i, char in enumerate(text):
        if char not in ALLOWED_CHARS:
            invalid_chars.add((char, i))
    
    if invalid_chars:
        sample = list(invalid_chars)[:10]
        raise ValueError(
            f"Invalid characters found in {filename}:\n"
            f"Found {len(invalid_chars)} invalid character(s). First 10: {sample}\n"
            f"Allowed characters: digits 0-9, +, -, H, B, newline"
        )
    return True


def load_all_data(data_dir=None, validate=True):
    """
    Load all text data from the data directory.
    
    Returns:
        str: Concatenated text from all files
        int: Total number of characters
    """
    if data_dir is None:
        data_dir = get_data_dir()
    
    # Find all .txt files
    txt_files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    
    if not txt_files:
        raise FileNotFoundError(
            f"No .txt files found in {data_dir}\n"
            f"Please place your training data files there."
        )
    
    print(f"Found {len(txt_files)} data file(s) in {data_dir}")
    
    all_text = []
    total_chars = 0
    
    for filepath in txt_files:
        filename = os.path.basename(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        if validate:
            validate_text(text, filename)
        
        all_text.append(text)
        total_chars += len(text)
        print(f"  - {filename}: {len(text):,} characters")
    
    combined = "\n".join(all_text)  # Join files with newlines
    print(f"Total: {total_chars:,} characters")
    
    return combined, total_chars


def data_iter_batched(split, data_dir=None, doc_separator="\n\n"):
    """
    Iterate through the dataset, yielding batches of documents.
    
    Args:
        split: "train" or "val" - uses 95%/5% split
        data_dir: Optional custom data directory
        doc_separator: String that separates documents (default: double newline)
    
    Yields:
        list[str]: Batches of document texts
    """
    text, _ = load_all_data(data_dir, validate=True)
    
    # Split into documents
    documents = text.split(doc_separator)
    documents = [doc.strip() for doc in documents if doc.strip()]
    
    # Split into train/val (95%/5%)
    split_idx = int(len(documents) * 0.95)
    if split == "train":
        documents = documents[:split_idx]
    else:
        documents = documents[split_idx:]
    
    print(f"Split '{split}': {len(documents):,} documents")
    
    # Yield in batches
    batch_size = 1024  # Number of documents per batch
    for i in range(0, len(documents), batch_size):
        yield documents[i:i + batch_size]


def get_data_stats(data_dir=None):
    """Get statistics about the dataset."""
    text, total_chars = load_all_data(data_dir, validate=True)
    
    # Character frequency
    char_counts = {}
    for char in text:
        char_counts[char] = char_counts.get(char, 0) + 1
    
    # Document count (assuming \n\n separates documents)
    documents = text.split("\n\n")
    documents = [d.strip() for d in documents if d.strip()]
    
    stats = {
        "total_characters": total_chars,
        "total_documents": len(documents),
        "character_counts": char_counts,
        "avg_doc_length": total_chars / len(documents) if documents else 0,
    }
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Dataset utilities for experiment")
    parser.add_argument("--stats", action="store_true", help="Show dataset statistics")
    parser.add_argument("--validate", action="store_true", help="Validate dataset")
    parser.add_argument("--sample", type=int, default=0, help="Show N sample documents")
    args = parser.parse_args()
    
    data_dir = get_data_dir()
    print(f"Data directory: {data_dir}")
    print()
    
    if args.validate or args.stats:
        try:
            stats = get_data_stats()
            print("\nDataset Statistics:")
            print(f"  Total characters: {stats['total_characters']:,}")
            print(f"  Total documents: {stats['total_documents']:,}")
            print(f"  Avg doc length: {stats['avg_doc_length']:.1f} chars")
            print(f"\nCharacter frequencies:")
            for char, count in sorted(stats['character_counts'].items()):
                char_repr = repr(char) if char != '\n' else "'\\n'"
                print(f"    {char_repr}: {count:,} ({100*count/stats['total_characters']:.2f}%)")
        except Exception as e:
            print(f"Error: {e}")
    
    if args.sample > 0:
        print(f"\nFirst {args.sample} documents:")
        for i, batch in enumerate(data_iter_batched("train")):
            for j, doc in enumerate(batch[:args.sample]):
                print(f"\n--- Document {i*1024 + j + 1} ---")
                print(doc[:500] + ("..." if len(doc) > 500 else ""))
            break
