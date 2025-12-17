#!/usr/bin/env python3
"""
Script to count the number of samples in the dataset directory.
Counts structures in *_train.xyz and *_validation.xyz files.
"""

import sys
from pathlib import Path


def count_structures_in_file(filepath: Path) -> int:
    """Count the number of structures in a single XYZ file.
    
    XYZ format: Each structure starts with a line containing the number of atoms,
    followed by a comment line, then that many atom lines.
    """
    count = 0
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            # Check if this line is a number (atom count)
            if line.isdigit():
                num_atoms = int(line)
                count += 1
                # Skip the comment line and atom lines
                i += 1  # comment line
                i += num_atoms  # atom lines
            else:
                i += 1
    except Exception as e:
        print(f"Warning: Could not read {filepath}: {e}")
        return 0
    
    return count


def count_samples(data_path: str):
    """Count samples in train and validation files."""
    data_dir = Path(data_path)
    
    if not data_dir.exists():
        print(f"Error: Directory does not exist: {data_path}")
        sys.exit(1)
    
    train_files = list(data_dir.glob("*_train.xyz"))
    val_files = list(data_dir.glob("*_validation.xyz"))
    
    print(f"Found {len(train_files)} train file(s)")
    print(f"Found {len(val_files)} validation file(s)")
    print()
    
    train_total = 0
    val_total = 0
    
    print("Counting train samples...")
    for train_file in train_files:
        count = count_structures_in_file(train_file)
        train_total += count
        print(f"  {train_file.name}: {count} structures")
    
    print()
    print("Counting validation samples...")
    for val_file in val_files:
        count = count_structures_in_file(val_file)
        val_total += count
        print(f"  {val_file.name}: {count} structures")
    
    print()
    print("=" * 50)
    print(f"Total train samples: {train_total}")
    print(f"Total validation samples: {val_total}")
    print(f"Total samples: {train_total + val_total}")
    print("=" * 50)

"""
uv run scripts/count_samples.py data/300/1090
uv run scripts/count_samples.py data/all/8020
"""
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run scripts/count_samples.py <data_directory>")
        print("Example: uv run scripts/count_samples.py data/300/1090")
        sys.exit(1)
    
    count_samples(sys.argv[1])

