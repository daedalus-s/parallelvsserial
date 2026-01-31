#!/usr/bin/env python3
"""
Data File Generator for Max Finder Benchmark

Generates binary files containing random double-precision floating point numbers.
Optimized for fast generation using numpy.
"""

import argparse
import os
import sys
import time
from typing import Optional

import numpy as np


def generate_data_file(filepath: str, size_gb: float, 
                       chunk_size_mb: int = 256,
                       seed: Optional[int] = None,
                       known_max: Optional[float] = None,
                       max_position: str = 'random') -> dict:
    """
    Generate a binary file with random double-precision floats.
    
    Args:
        filepath: Output file path
        size_gb: Target file size in gigabytes
        chunk_size_mb: Size of each write chunk in MB
        seed: Random seed for reproducibility
        known_max: If set, insert this known maximum value
        max_position: Where to insert known_max ('start', 'middle', 'end', 'random')
    
    Returns:
        Dictionary with generation statistics
    """
    if seed is not None:
        np.random.seed(seed)
    
    double_size = 8  # 8 bytes per double
    total_bytes = int(size_gb * 1024 * 1024 * 1024)
    chunk_bytes = chunk_size_mb * 1024 * 1024
    
    # Ensure chunk is aligned to double size
    chunk_bytes = (chunk_bytes // double_size) * double_size
    doubles_per_chunk = chunk_bytes // double_size
    
    total_doubles = total_bytes // double_size
    num_chunks = (total_doubles + doubles_per_chunk - 1) // doubles_per_chunk
    
    # Determine where to place the known maximum
    max_chunk_idx = None
    max_position_in_chunk = None
    actual_max = None
    
    if known_max is not None:
        if max_position == 'start':
            max_chunk_idx = 0
            max_position_in_chunk = 0
        elif max_position == 'middle':
            max_chunk_idx = num_chunks // 2
            max_position_in_chunk = doubles_per_chunk // 2
        elif max_position == 'end':
            max_chunk_idx = num_chunks - 1
            max_position_in_chunk = -1  # Will be set based on actual chunk size
        else:  # random
            max_chunk_idx = np.random.randint(0, num_chunks)
            max_position_in_chunk = np.random.randint(0, doubles_per_chunk)
        actual_max = known_max
    
    print(f"\n{'='*60}")
    print(f"Data File Generator")
    print(f"{'='*60}")
    print(f"Output: {filepath}")
    print(f"Target Size: {size_gb:.2f} GB ({total_bytes:,} bytes)")
    print(f"Number of doubles: {total_doubles:,}")
    print(f"Chunk Size: {chunk_size_mb} MB ({doubles_per_chunk:,} doubles/chunk)")
    print(f"Number of chunks: {num_chunks}")
    if known_max is not None:
        print(f"Known Max: {known_max} (position: {max_position})")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    bytes_written = 0
    observed_max = float('-inf')
    
    with open(filepath, 'wb') as f:
        for chunk_idx in range(num_chunks):
            # Calculate how many doubles for this chunk
            remaining_doubles = total_doubles - (chunk_idx * doubles_per_chunk)
            current_chunk_doubles = min(doubles_per_chunk, remaining_doubles)
            
            # Generate random doubles between -1e15 and 1e15
            # Using a range that's realistic but won't overflow
            data = np.random.uniform(-1e15, 1e15, current_chunk_doubles)
            
            # Insert known maximum if this is the designated chunk
            if chunk_idx == max_chunk_idx and known_max is not None:
                pos = max_position_in_chunk
                if pos == -1:
                    pos = current_chunk_doubles - 1
                elif pos >= current_chunk_doubles:
                    pos = current_chunk_doubles - 1
                data[pos] = known_max
            
            # Track observed maximum
            chunk_max = np.max(data)
            if chunk_max > observed_max:
                observed_max = chunk_max
            
            # Write to file
            f.write(data.tobytes())
            bytes_written += current_chunk_doubles * double_size
            
            # Progress update
            progress = (chunk_idx + 1) / num_chunks * 100
            elapsed = time.time() - start_time
            speed = bytes_written / (1024 * 1024) / elapsed if elapsed > 0 else 0
            
            print(f"\r[{progress:5.1f}%] Written: {bytes_written / (1024**3):.2f} GB | "
                  f"Speed: {speed:.0f} MB/s", end='', flush=True)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    actual_size = os.path.getsize(filepath)
    
    print(f"\n\n{'='*60}")
    print(f"Generation Complete")
    print(f"{'='*60}")
    print(f"File Size: {actual_size / (1024**3):.2f} GB ({actual_size:,} bytes)")
    print(f"Time: {elapsed:.1f} seconds")
    print(f"Speed: {actual_size / (1024*1024) / elapsed:.0f} MB/s")
    print(f"Observed Max: {observed_max}")
    if known_max is not None:
        print(f"Known Max: {known_max}")
    print(f"{'='*60}\n")
    
    return {
        'filepath': filepath,
        'size_bytes': actual_size,
        'elapsed_time': elapsed,
        'observed_max': observed_max,
        'known_max': known_max,
        'num_doubles': total_doubles
    }


def main():
    parser = argparse.ArgumentParser(
        description='Generate binary data files for max finder benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 1 GB test file
  python generate_data.py test_1gb.bin --size 1
  
  # Generate 200 GB file
  python generate_data.py data_200gb.bin --size 200
  
  # Generate with known maximum for verification
  python generate_data.py test.bin --size 10 --known-max 9.99e99
  
  # Generate reproducible data
  python generate_data.py test.bin --size 5 --seed 42
        """
    )
    
    parser.add_argument('filepath', help='Output file path')
    parser.add_argument('-s', '--size', type=float, required=True,
                        help='File size in gigabytes')
    parser.add_argument('-c', '--chunk-size', type=int, default=256,
                        help='Write chunk size in MB (default: 256)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--known-max', type=float, default=None,
                        help='Insert a known maximum value for verification')
    parser.add_argument('--max-position', choices=['start', 'middle', 'end', 'random'],
                        default='random',
                        help='Position for known max (default: random)')
    
    args = parser.parse_args()
    
    if args.size <= 0:
        print("Error: Size must be positive", file=sys.stderr)
        sys.exit(1)
    
    # Check available disk space
    try:
        stat = os.statvfs(os.path.dirname(os.path.abspath(args.filepath)) or '.')
        available_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        if available_gb < args.size * 1.1:  # 10% buffer
            print(f"Warning: Low disk space. Available: {available_gb:.1f} GB, "
                  f"Requested: {args.size:.1f} GB")
            response = input("Continue anyway? [y/N] ")
            if response.lower() != 'y':
                sys.exit(0)
    except Exception:
        pass  # Ignore if we can't check disk space
    
    try:
        results = generate_data_file(
            args.filepath,
            args.size,
            args.chunk_size,
            args.seed,
            args.known_max,
            args.max_position
        )
        
        print(f"Successfully generated {args.filepath}")
        print(f"To verify: python max_finder.py {args.filepath}")
        
    except KeyboardInterrupt:
        print("\n\nGeneration interrupted by user")
        # Clean up partial file
        if os.path.exists(args.filepath):
            os.remove(args.filepath)
            print(f"Removed partial file: {args.filepath}")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
