#!/usr/bin/env python3
"""
Fastest Possible Data Generator

Supports two output formats:
1. 'binary' - Binary file with double-precision floats (fast I/O, compact)
2. 'text' - Text file with comma-separated integers (human-readable)

Generation modes:
1. 'random' - Full random data
2. 'fast-random' - Faster RNG (PCG64 DXSM)
3. 'pattern' - Repeating pattern (fastest)
"""

import argparse
import multiprocessing as mp
import numpy as np
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import psutil


class IOWaitMonitor:
    """Monitor I/O wait time by tracking wall time vs CPU time."""
    
    def __init__(self):
        self.wall_start = 0.0
        self.cpu_start = 0.0
        self.process = psutil.Process()
        
    def start(self):
        self.wall_start = time.time()
        cpu_times = self.process.cpu_times()
        self.cpu_start = cpu_times.user + cpu_times.system
        
    def get_stats(self) -> dict:
        wall_elapsed = time.time() - self.wall_start
        cpu_times = self.process.cpu_times()
        cpu_elapsed = (cpu_times.user + cpu_times.system) - self.cpu_start
        io_wait = max(0, wall_elapsed - cpu_elapsed)
        io_wait_pct = (io_wait / wall_elapsed * 100) if wall_elapsed > 0 else 0
        
        return {
            'wall_time': wall_elapsed,
            'cpu_time': cpu_elapsed,
            'io_wait_time': io_wait,
            'io_wait_pct': io_wait_pct,
            'user_time': cpu_times.user,
            'system_time': cpu_times.system
        }


# =============================================================================
# TEXT FORMAT GENERATORS (Comma-separated integers)
# =============================================================================

def write_text_segment_random(args):
    """Write segment with random integers as comma-separated text."""
    filepath, segment_id, num_integers, seed, known_max_info, is_first_segment, is_last_segment = args
    
    rng = np.random.Generator(np.random.PCG64(seed + segment_id * 7919))
    
    # Generate in chunks to manage memory
    chunk_size = min(num_integers, 10_000_000)  # 10M integers per chunk
    written = 0
    bytes_written = 0
    
    # Open in append mode
    with open(filepath, 'a') as f:
        while written < num_integers:
            n = min(chunk_size, num_integers - written)
            
            # Generate random integers (range: -10^15 to 10^15)
            data = rng.integers(-10**15, 10**15, n, dtype=np.int64)
            
            # Insert known max
            if known_max_info and written == 0:
                max_val, pos = known_max_info
                if pos < n:
                    data[pos] = int(max_val)
            
            # Convert to comma-separated string
            # Add leading comma if not first chunk of first segment
            if written > 0 or not is_first_segment:
                text = ',' + ','.join(map(str, data))
            else:
                text = ','.join(map(str, data))
            
            f.write(text)
            bytes_written += len(text)
            written += n
    
    return segment_id, bytes_written


def write_text_segment_pattern(args):
    """Write segment with pattern integers as comma-separated text."""
    filepath, segment_id, num_integers, seed, known_max_info, is_first_segment, is_last_segment = args
    
    chunk_size = min(num_integers, 10_000_000)
    written = 0
    bytes_written = 0
    
    with open(filepath, 'a') as f:
        while written < num_integers:
            n = min(chunk_size, num_integers - written)
            
            # Create pattern: segment_id * 1000000 + index
            base = np.arange(written, written + n, dtype=np.int64)
            data = (base + segment_id * 100_000_000) % (10**15)
            
            # Insert known max
            if known_max_info and written == 0:
                max_val, _ = known_max_info
                data[0] = int(max_val)
            
            # Add leading comma if not first
            if written > 0 or not is_first_segment:
                text = ',' + ','.join(map(str, data))
            else:
                text = ','.join(map(str, data))
            
            f.write(text)
            bytes_written += len(text)
            written += n
    
    return segment_id, bytes_written


def generate_text_file(filepath: str, size_gb: float, mode: str = 'random',
                       num_workers: int = None, seed: int = None,
                       known_max: int = None, max_position: str = 'random') -> dict:
    """
    Generate a text file with comma-separated integers.
    
    Note: Text files are larger than binary for the same number of values.
    A 1GB text file contains roughly 50-60 million integers.
    """
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    if seed is None:
        seed = 42
    
    # Estimate: average integer string is ~16 chars + comma = ~17 bytes
    # So 1GB text ≈ 60 million integers
    avg_bytes_per_int = 17
    total_bytes_target = int(size_gb * 1024 * 1024 * 1024)
    total_integers = total_bytes_target // avg_bytes_per_int
    
    # Round to make division even
    total_integers = (total_integers // num_workers) * num_workers
    integers_per_segment = total_integers // num_workers
    
    print(f"\n{'='*70}")
    print(f"Text Data Generator - Mode: {mode.upper()}")
    print(f"{'='*70}")
    print(f"Output: {filepath}")
    print(f"Target Size: ~{size_gb:.2f} GB")
    print(f"Estimated Integers: {total_integers:,}")
    print(f"Workers: {num_workers}")
    print(f"Format: Comma-separated integers (text)")
    if known_max:
        print(f"Known Max: {known_max}")
    print(f"{'='*70}\n")
    
    # Determine known max position
    max_segment = None
    max_pos_in_segment = None
    if known_max is not None:
        rng = np.random.default_rng(seed)
        if max_position == 'start':
            global_pos = 0
        elif max_position == 'middle':
            global_pos = total_integers // 2
        elif max_position == 'end':
            global_pos = total_integers - 1
        else:
            global_pos = rng.integers(0, total_integers)
        
        max_segment = global_pos // integers_per_segment
        max_pos_in_segment = global_pos % integers_per_segment
    
    # Create empty file
    print("Creating file...")
    with open(filepath, 'w') as f:
        pass  # Create empty file
    
    # For text files, we need to write segments sequentially to maintain order
    # But we can generate the content in parallel
    print(f"Generating {total_integers:,} integers...")
    
    io_monitor = IOWaitMonitor()
    io_monitor.start()
    psutil.cpu_percent(percpu=True)
    cpu_samples = []
    
    start_time = time.time()
    total_bytes_written = 0
    
    # Select worker function
    if mode == 'pattern':
        worker_fn = write_text_segment_pattern
    else:
        worker_fn = write_text_segment_random
    
    # Process segments sequentially (required for text format to maintain order)
    for i in range(num_workers):
        known_max_info = None
        if i == max_segment:
            known_max_info = (known_max, max_pos_in_segment)
        
        is_first = (i == 0)
        is_last = (i == num_workers - 1)
        
        args = (filepath, i, integers_per_segment, seed, known_max_info, is_first, is_last)
        seg_id, seg_bytes = worker_fn(args)
        
        total_bytes_written += seg_bytes
        
        cpu_pcts = psutil.cpu_percent(percpu=True)
        cpu_samples.append(cpu_pcts)
        
        elapsed = time.time() - start_time
        speed = total_bytes_written / (1024**2) / elapsed if elapsed > 0 else 0
        pct = (i + 1) / num_workers * 100
        
        print(f"\r[{pct:5.1f}%] {total_bytes_written/(1024**3):.2f} GB | {speed:.0f} MB/s", 
              end='', flush=True)
    
    elapsed = time.time() - start_time
    actual_size = os.path.getsize(filepath)
    speed = actual_size / (1024**2) / elapsed if elapsed > 0 else 0
    
    io_stats = io_monitor.get_stats()
    
    # Calculate per-core averages
    if cpu_samples:
        num_cores = len(cpu_samples[0])
        per_core_avg = [sum(s[i] for s in cpu_samples) / len(cpu_samples) for i in range(num_cores)]
        active_cores = [i for i, avg in enumerate(per_core_avg) if avg > 10.0]
    else:
        per_core_avg = []
        active_cores = []
    
    print(f"\n\n{'='*70}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"File Size: {actual_size/(1024**3):.2f} GB ({actual_size:,} bytes)")
    print(f"Total Integers: {total_integers:,}")
    print(f"Total Time: {elapsed:.1f} seconds")
    print(f"Write Speed: {speed:.0f} MB/s")
    
    print(f"\n--- I/O Wait Time ---")
    print(f"Wall Clock Time: {io_stats['wall_time']:.2f} seconds")
    print(f"CPU Time: {io_stats['cpu_time']:.2f} seconds")
    print(f"I/O Wait (est): {io_stats['io_wait_time']:.2f} seconds ({io_stats['io_wait_pct']:.1f}%)")
    
    if per_core_avg:
        print(f"\n--- CPU Cores Used ---")
        print(f"Total Cores: {len(per_core_avg)}")
        print(f"Active Cores: {len(active_cores)}")
    
    print(f"{'='*70}\n")
    
    return {
        'filepath': filepath,
        'size': actual_size,
        'num_integers': total_integers,
        'time': elapsed,
        'speed_mbps': speed,
        'io_stats': io_stats
    }


# =============================================================================
# BINARY FORMAT GENERATORS (Double-precision floats)
# =============================================================================

def write_segment_random(args):
    """Write segment with full random data (binary doubles)."""
    filepath, segment_id, start_byte, num_doubles, seed, known_max_info = args
    
    rng = np.random.Generator(np.random.PCG64(seed + segment_id * 7919))
    
    with open(filepath, 'r+b') as f:
        f.seek(start_byte)
        
        chunk_size = 128 * 1024 * 1024 // 8
        written = 0
        
        while written < num_doubles:
            n = min(chunk_size, num_doubles - written)
            data = rng.uniform(-1e15, 1e15, n)
            
            if known_max_info and written == 0:
                max_val, pos = known_max_info
                if pos < n:
                    data[pos] = max_val
            
            data.tofile(f)
            written += n
    
    return segment_id, num_doubles * 8


def write_segment_fast_random(args):
    """Write segment with fast RNG (binary doubles)."""
    filepath, segment_id, start_byte, num_doubles, seed, known_max_info = args
    
    rng = np.random.Generator(np.random.PCG64DXSM(seed + segment_id * 7919))
    
    with open(filepath, 'r+b') as f:
        f.seek(start_byte)
        
        chunk_size = 128 * 1024 * 1024 // 8
        written = 0
        
        while written < num_doubles:
            n = min(chunk_size, num_doubles - written)
            data = rng.standard_normal(n) * 1e14
            
            if known_max_info and written == 0:
                max_val, pos = known_max_info
                if pos < n:
                    data[pos] = max_val
            
            data.tofile(f)
            written += n
    
    return segment_id, num_doubles * 8


def write_segment_pattern(args):
    """Write segment with repeating pattern (binary doubles)."""
    filepath, segment_id, start_byte, num_doubles, seed, known_max_info = args
    
    with open(filepath, 'r+b') as f:
        f.seek(start_byte)
        
        pattern_size = 256 * 1024 * 1024 // 8
        base = np.arange(pattern_size, dtype=np.float64)
        pattern = (base * 1.23456789 + segment_id * 1e10) % 1e15 - 5e14
        
        if known_max_info:
            max_val, _ = known_max_info
            pattern[0] = max_val
        
        written = 0
        while written < num_doubles:
            n = min(pattern_size, num_doubles - written)
            pattern[:n].tofile(f)
            written += n
    
    return segment_id, num_doubles * 8


def generate_binary_file(filepath: str, size_gb: float, mode: str = 'random',
                         num_workers: int = None, seed: int = None,
                         known_max: float = None, max_position: str = 'random') -> dict:
    """Generate a binary file with double-precision floats."""
    
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    if seed is None:
        seed = 42
    
    double_size = 8
    total_bytes = int(size_gb * 1024 * 1024 * 1024)
    total_doubles = total_bytes // double_size
    
    total_doubles = (total_doubles // num_workers) * num_workers
    total_bytes = total_doubles * double_size
    doubles_per_segment = total_doubles // num_workers
    
    print(f"\n{'='*70}")
    print(f"Binary Data Generator - Mode: {mode.upper()}")
    print(f"{'='*70}")
    print(f"Output: {filepath}")
    print(f"Size: {total_bytes / (1024**3):.2f} GB")
    print(f"Workers: {num_workers}")
    print(f"Format: Binary (double-precision floats)")
    if known_max:
        print(f"Known Max: {known_max}")
    print(f"{'='*70}\n")
    
    # Determine known max position
    max_segment = None
    max_pos_in_segment = None
    if known_max is not None:
        rng = np.random.default_rng(seed)
        if max_position == 'start':
            global_pos = 0
        elif max_position == 'middle':
            global_pos = total_doubles // 2
        elif max_position == 'end':
            global_pos = total_doubles - 1
        else:
            global_pos = rng.integers(0, total_doubles)
        
        max_segment = global_pos // doubles_per_segment
        max_pos_in_segment = global_pos % doubles_per_segment
    
    print("Pre-allocating file...")
    with open(filepath, 'wb') as f:
        f.truncate(total_bytes)
    
    # Build tasks
    tasks = []
    for i in range(num_workers):
        start_byte = i * doubles_per_segment * double_size
        known_max_info = None
        if i == max_segment:
            known_max_info = (known_max, max_pos_in_segment)
        tasks.append((filepath, i, start_byte, doubles_per_segment, seed, known_max_info))
    
    # Select worker function
    if mode == 'pattern':
        worker_fn = write_segment_pattern
    elif mode == 'fast-random':
        worker_fn = write_segment_fast_random
    else:
        worker_fn = write_segment_random
    
    print(f"Starting {num_workers} workers...")
    
    io_monitor = IOWaitMonitor()
    io_monitor.start()
    psutil.cpu_percent(percpu=True)
    cpu_samples = []
    
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker_fn, task) for task in tasks]
        
        completed = 0
        bytes_done = 0
        
        for future in as_completed(futures):
            seg_id, seg_bytes = future.result()
            completed += 1
            bytes_done += seg_bytes
            
            cpu_pcts = psutil.cpu_percent(percpu=True)
            cpu_samples.append(cpu_pcts)
            
            elapsed = time.time() - start_time
            speed = bytes_done / (1024**2) / elapsed
            pct = completed / num_workers * 100
            
            print(f"\r[{pct:5.1f}%] {bytes_done/(1024**3):.2f} GB | {speed:.0f} MB/s", 
                  end='', flush=True)
    
    elapsed = time.time() - start_time
    actual_size = os.path.getsize(filepath)
    speed = actual_size / (1024**2) / elapsed
    
    io_stats = io_monitor.get_stats()
    
    if cpu_samples:
        num_cores = len(cpu_samples[0])
        per_core_avg = [sum(s[i] for s in cpu_samples) / len(cpu_samples) for i in range(num_cores)]
        active_cores = [i for i, avg in enumerate(per_core_avg) if avg > 10.0]
    else:
        per_core_avg = []
        active_cores = []
    
    print(f"\n\n{'='*70}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"File Size: {actual_size/(1024**3):.2f} GB")
    print(f"Total Time: {elapsed:.1f} seconds")
    print(f"Write Speed: {speed:.0f} MB/s ({speed/1024:.2f} GB/s)")
    
    print(f"\n--- I/O Wait Time ---")
    print(f"Wall Clock Time: {io_stats['wall_time']:.2f} seconds")
    print(f"CPU Time: {io_stats['cpu_time']:.2f} seconds")
    print(f"I/O Wait (est): {io_stats['io_wait_time']:.2f} seconds ({io_stats['io_wait_pct']:.1f}%)")
    
    if per_core_avg:
        print(f"\n--- CPU Cores Used ---")
        print(f"Total Cores: {len(per_core_avg)}")
        print(f"Active Cores: {len(active_cores)}")
        if active_cores:
            print(f"Active Core IDs: {', '.join(f'Core {c}' for c in active_cores)}")
        
        print(f"\nPer-Core Utilization:")
        cols = 4
        for i in range(0, len(per_core_avg), cols):
            row = []
            for j in range(cols):
                if i + j < len(per_core_avg):
                    core_id = i + j
                    pct = per_core_avg[core_id]
                    marker = "*" if core_id in active_cores else " "
                    row.append(f"Core {core_id:2d}:{marker}{pct:5.1f}%")
            print("  " + "  |  ".join(row))
        print(f"  (* = active core)")
    
    print(f"{'='*70}\n")
    
    return {
        'filepath': filepath,
        'size': actual_size,
        'time': elapsed,
        'speed_mbps': speed,
        'io_stats': io_stats,
        'active_cores': active_cores
    }


def main():
    parser = argparse.ArgumentParser(
        description='Fast data generator with text and binary format support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Formats:
  text   - Comma-separated integers (human-readable)
  binary - Double-precision floats (compact, fast)

Modes:
  random      - Full quality random numbers
  fast-random - Faster RNG (binary only)
  pattern     - Repeating pattern (fastest)

Examples:
  # Generate 1GB text file with comma-separated integers
  python generate_fastest.py data.txt --size 1 --format text
  
  # Generate 1GB binary file (default)
  python generate_fastest.py data.bin --size 1 --format binary
  
  # With known maximum for verification
  python generate_fastest.py data.txt --size 1 --format text --known-max 999999999999999
        """
    )
    
    parser.add_argument('filepath', help='Output path')
    parser.add_argument('-s', '--size', type=float, required=True, help='Size in GB')
    parser.add_argument('-f', '--format', choices=['text', 'binary'], default='text',
                        help='Output format (default: text)')
    parser.add_argument('-m', '--mode', choices=['random', 'fast-random', 'pattern'],
                        default='random', help='Generation mode')
    parser.add_argument('-w', '--workers', type=int, default=None, help='Worker count')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--known-max', type=float, help='Insert known max value')
    parser.add_argument('--max-position', choices=['start', 'middle', 'end', 'random'],
                        default='random', help='Position for known max')
    
    args = parser.parse_args()
    
    # Check disk space
    try:
        stat = os.statvfs(os.path.dirname(args.filepath) or '.')
        avail = stat.f_bavail * stat.f_frsize / (1024**3)
        if avail < args.size * 1.1:
            print(f"Error: Need ~{args.size:.1f}GB, have {avail:.1f}GB")
            sys.exit(1)
    except:
        pass
    
    try:
        if args.format == 'text':
            # For text format, known_max should be an integer
            known_max = int(args.known_max) if args.known_max else None
            result = generate_text_file(
                args.filepath, args.size, args.mode, args.workers,
                args.seed, known_max, args.max_position
            )
            print(f"Generated text file: {args.filepath}")
            print(f"Format: Comma-separated integers")
            print(f"To read: Use max_finder_text.py or parse as CSV")
        else:
            result = generate_binary_file(
                args.filepath, args.size, args.mode, args.workers,
                args.seed, args.known_max, args.max_position
            )
            print(f"Verify: python max_finder.py {args.filepath} --threads 0")
            
    except KeyboardInterrupt:
        print("\nCancelled")
        if os.path.exists(args.filepath):
            os.remove(args.filepath)
        sys.exit(130)


if __name__ == '__main__':
    main()
