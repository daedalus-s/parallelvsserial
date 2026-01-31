#!/usr/bin/env python3
"""
Comprehensive Benchmark Script

Compares single-threaded vs multi-threaded performance
with various configurations.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import psutil

from max_finder import run_benchmark


def run_full_benchmark(filepath: str, thread_counts: list[int], 
                       chunk_sizes: list[int], runs_per_config: int = 1) -> dict:
    """
    Run comprehensive benchmark with multiple configurations.
    
    Args:
        filepath: Path to test data file
        thread_counts: List of thread counts to test
        chunk_sizes: List of chunk sizes in MB to test
        runs_per_config: Number of runs per configuration
    
    Returns:
        Dictionary with all benchmark results
    """
    file_size = os.path.getsize(filepath)
    num_cores = psutil.cpu_count(logical=True)
    
    print(f"\n{'#'*70}")
    print(f"# COMPREHENSIVE BENCHMARK")
    print(f"{'#'*70}")
    print(f"File: {filepath}")
    print(f"File Size: {file_size / (1024**3):.2f} GB")
    print(f"System Cores: {num_cores} (logical)")
    print(f"Thread Counts: {thread_counts}")
    print(f"Chunk Sizes: {chunk_sizes} MB")
    print(f"Runs per Config: {runs_per_config}")
    print(f"{'#'*70}\n")
    
    results = {
        'metadata': {
            'filepath': filepath,
            'file_size_bytes': file_size,
            'system_cores': num_cores,
            'timestamp': datetime.now().isoformat(),
            'platform': sys.platform
        },
        'configurations': []
    }
    
    total_configs = len(thread_counts) * len(chunk_sizes) * runs_per_config
    current = 0
    
    for chunk_size in chunk_sizes:
        for num_threads in thread_counts:
            for run in range(runs_per_config):
                current += 1
                print(f"\n[Config {current}/{total_configs}] "
                      f"Threads: {num_threads}, Chunk: {chunk_size} MB, Run: {run + 1}")
                
                try:
                    result = run_benchmark(filepath, num_threads, chunk_size)
                    
                    config_result = {
                        'num_threads': num_threads,
                        'chunk_size_mb': chunk_size,
                        'run_number': run + 1,
                        'elapsed_time': result['elapsed_time'],
                        'throughput_avg': result['throughput']['avg_throughput'],
                        'throughput_peak': result['throughput']['peak_throughput'],
                        'cpu_avg': result['cpu']['overall_avg'],
                        'max_value': result['max_value']
                    }
                    results['configurations'].append(config_result)
                    
                    # Brief pause between runs to let system settle
                    if current < total_configs:
                        print("Cooling down for 2 seconds...")
                        time.sleep(2)
                        
                except Exception as e:
                    print(f"Error in configuration: {e}")
                    results['configurations'].append({
                        'num_threads': num_threads,
                        'chunk_size_mb': chunk_size,
                        'run_number': run + 1,
                        'error': str(e)
                    })
    
    return results


def print_summary(results: dict):
    """Print a formatted summary of benchmark results."""
    
    print(f"\n{'='*70}")
    print(f"BENCHMARK SUMMARY")
    print(f"{'='*70}")
    
    # Group by thread count
    by_threads = {}
    for config in results['configurations']:
        if 'error' in config:
            continue
        threads = config['num_threads']
        if threads not in by_threads:
            by_threads[threads] = []
        by_threads[threads].append(config)
    
    # Find best configurations
    configs_no_error = [c for c in results['configurations'] if 'error' not in c]
    if configs_no_error:
        best_time = min(configs_no_error, key=lambda x: x['elapsed_time'])
        best_throughput = max(configs_no_error, key=lambda x: x['throughput_avg'])
        single_thread = [c for c in configs_no_error if c['num_threads'] == 1]
        
        print(f"\nFastest Configuration:")
        print(f"  Threads: {best_time['num_threads']}, "
              f"Chunk: {best_time['chunk_size_mb']} MB")
        print(f"  Time: {best_time['elapsed_time']:.2f}s, "
              f"Throughput: {best_time['throughput_avg']:.0f} MB/s")
        
        print(f"\nBest Throughput:")
        print(f"  Threads: {best_throughput['num_threads']}, "
              f"Chunk: {best_throughput['chunk_size_mb']} MB")
        print(f"  Throughput: {best_throughput['throughput_avg']:.0f} MB/s (peak: "
              f"{best_throughput['throughput_peak']:.0f} MB/s)")
        
        if single_thread:
            baseline = single_thread[0]['elapsed_time']
            speedup = baseline / best_time['elapsed_time']
            print(f"\nSpeedup vs Single-Thread:")
            print(f"  Single-thread baseline: {baseline:.2f}s")
            print(f"  Best multi-thread: {best_time['elapsed_time']:.2f}s")
            print(f"  Speedup: {speedup:.2f}x")
    
    # Comparison table
    print(f"\n{'='*70}")
    print(f"Configuration Comparison")
    print(f"{'='*70}")
    print(f"{'Threads':>8} | {'Chunk MB':>10} | {'Time (s)':>10} | "
          f"{'Throughput':>12} | {'CPU %':>8}")
    print(f"{'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*8}")
    
    for config in sorted(configs_no_error, 
                        key=lambda x: (x['num_threads'], x['chunk_size_mb'])):
        print(f"{config['num_threads']:>8} | {config['chunk_size_mb']:>10} | "
              f"{config['elapsed_time']:>10.2f} | "
              f"{config['throughput_avg']:>9.0f} MB/s | "
              f"{config['cpu_avg']:>7.1f}%")
    
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run comprehensive benchmark comparing configurations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with default settings
  python benchmark.py data.bin
  
  # Test specific thread counts
  python benchmark.py data.bin --threads 1 2 4 8
  
  # Test with multiple chunk sizes
  python benchmark.py data.bin --chunk-sizes 32 64 128
  
  # Full benchmark with multiple runs
  python benchmark.py data.bin --threads 1 2 4 8 --chunk-sizes 32 64 --runs 3
  
  # Save results to JSON
  python benchmark.py data.bin --output results.json
        """
    )
    
    parser.add_argument('filepath', help='Path to test data file')
    parser.add_argument('-t', '--threads', type=int, nargs='+', 
                        default=[1, 2, 4],
                        help='Thread counts to test (default: 1 2 4)')
    parser.add_argument('-c', '--chunk-sizes', type=int, nargs='+',
                        default=[64],
                        help='Chunk sizes in MB to test (default: 64)')
    parser.add_argument('-r', '--runs', type=int, default=1,
                        help='Number of runs per configuration (default: 1)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output JSON file for results')
    parser.add_argument('--all-cores', action='store_true',
                        help='Include test with all available cores')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.filepath):
        print(f"Error: File '{args.filepath}' not found", file=sys.stderr)
        sys.exit(1)
    
    thread_counts = list(args.threads)
    if args.all_cores:
        all_cores = psutil.cpu_count(logical=True)
        if all_cores not in thread_counts:
            thread_counts.append(all_cores)
    
    try:
        results = run_full_benchmark(
            args.filepath,
            thread_counts,
            args.chunk_sizes,
            args.runs
        )
        
        print_summary(results)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {args.output}")
        
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
