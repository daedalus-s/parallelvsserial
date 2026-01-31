#!/usr/bin/env python3
"""
Text File Maximum Number Finder
Reads comma-separated integers from a text file and finds the maximum.

Supports:
- Single-core single-threaded mode
- Multi-core multi-threaded mode
- Real-time throughput and CPU monitoring
"""

import argparse
import csv
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict

import psutil


@dataclass
class Stats:
    """Statistics for monitoring performance."""
    bytes_read: int = 0
    numbers_read: int = 0
    start_time: float = 0.0
    lock: threading.Lock = None
    
    def __post_init__(self):
        if self.lock is None:
            self.lock = threading.Lock()
    
    def add_progress(self, bytes_count: int, numbers_count: int):
        with self.lock:
            self.bytes_read += bytes_count
            self.numbers_read += numbers_count
    
    def get_throughput_mbps(self) -> float:
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            return (self.bytes_read / (1024 * 1024)) / elapsed
        return 0.0


class IOWaitMonitor:
    """Monitor I/O wait time."""
    
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
            'cpu_busy_pct': min(100, (cpu_elapsed / wall_elapsed * 100) if wall_elapsed > 0 else 0),
            'user_time': cpu_times.user - self.cpu_start,
            'system_time': cpu_times.system
        }


class CPUMonitor:
    """Monitor CPU utilization per core."""
    
    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.running = False
        self.thread = None
        self.cpu_samples = []
        self.cpu_times_start = []
        self.cpu_times_end = []
        self.start_time = 0.0
        self.end_time = 0.0
        
    def start(self):
        self.running = True
        self.cpu_samples = []
        self.start_time = time.time()
        self.cpu_times_start = psutil.cpu_times(percpu=True)
        psutil.cpu_percent(percpu=True)
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        self.end_time = time.time()
        self.cpu_times_end = psutil.cpu_times(percpu=True)
        if self.thread:
            self.thread.join(timeout=2.0)
    
    def _monitor_loop(self):
        while self.running:
            cpu_percents = psutil.cpu_percent(percpu=True, interval=self.interval)
            self.cpu_samples.append({
                'timestamp': time.time(),
                'per_core': cpu_percents,
                'average': sum(cpu_percents) / len(cpu_percents)
            })
    
    def get_per_core_times(self) -> List[Dict]:
        if not self.cpu_times_start or not self.cpu_times_end:
            return []
        
        per_core_times = []
        for i, (start, end) in enumerate(zip(self.cpu_times_start, self.cpu_times_end)):
            on_cpu_time = (end.user + end.system) - (start.user + start.system)
            idle_time = end.idle - start.idle
            total_time = on_cpu_time + idle_time
            on_cpu_pct = (on_cpu_time / total_time * 100) if total_time > 0 else 0
            
            per_core_times.append({
                'core_id': i,
                'on_cpu_time': on_cpu_time,
                'idle_time': idle_time,
                'on_cpu_pct': on_cpu_pct,
                'off_cpu_pct': 100 - on_cpu_pct
            })
        return per_core_times
    
    def get_summary(self) -> dict:
        if not self.cpu_samples:
            return {'per_core_avg': [], 'overall_avg': 0.0, 'cores_used': 0, 'total_cores': 0}
        
        num_cores = len(self.cpu_samples[0]['per_core'])
        per_core_totals = [0.0] * num_cores
        
        for sample in self.cpu_samples:
            for i, val in enumerate(sample['per_core']):
                per_core_totals[i] += val
        
        num_samples = len(self.cpu_samples)
        per_core_avg = [total / num_samples for total in per_core_totals]
        overall_avg = sum(per_core_avg) / len(per_core_avg)
        consistently_active = [i for i, avg in enumerate(per_core_avg) if avg > 10.0]
        
        return {
            'per_core_avg': per_core_avg,
            'overall_avg': overall_avg,
            'num_samples': num_samples,
            'consistently_active_cores': consistently_active,
            'cores_used': len(consistently_active),
            'total_cores': num_cores,
            'per_core_times': self.get_per_core_times()
        }


class ThroughputMonitor:
    """Monitor and display throughput in real-time."""
    
    def __init__(self, stats: Stats, interval: float = 1.0):
        self.stats = stats
        self.interval = interval
        self.running = False
        self.thread = None
        self.samples = []
    
    def start(self):
        self.running = True
        self.samples = []
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
    
    def _monitor_loop(self):
        last_bytes = 0
        last_time = time.time()
        
        while self.running:
            time.sleep(self.interval)
            current_bytes = self.stats.bytes_read
            current_time = time.time()
            
            delta_bytes = current_bytes - last_bytes
            delta_time = current_time - last_time
            
            if delta_time > 0:
                throughput_mbps = (delta_bytes / (1024 * 1024)) / delta_time
                self.samples.append(throughput_mbps)
                
                total_read_gb = current_bytes / (1024 * 1024 * 1024)
                numbers = self.stats.numbers_read
                print(f"\r[Progress] Read: {total_read_gb:.2f} GB | "
                      f"Numbers: {numbers:,} | "
                      f"Throughput: {throughput_mbps:.2f} MB/s", 
                      end='', flush=True)
            
            last_bytes = current_bytes
            last_time = current_time
    
    def get_summary(self) -> dict:
        if not self.samples:
            return {'avg_throughput': 0.0, 'peak_throughput': 0.0, 'min_throughput': 0.0}
        return {
            'avg_throughput': sum(self.samples) / len(self.samples),
            'peak_throughput': max(self.samples),
            'min_throughput': min(self.samples)
        }


def find_max_single_thread(filepath: str, chunk_size: int = 64 * 1024 * 1024,
                           stats: Optional[Stats] = None) -> int:
    """Find maximum integer in text file using single thread."""
    max_val = float('-inf')
    buffer = ""
    
    with open(filepath, 'r') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                # Process remaining buffer
                if buffer:
                    for num_str in buffer.split(','):
                        num_str = num_str.strip()
                        if num_str:
                            val = int(num_str)
                            if val > max_val:
                                max_val = val
                            if stats:
                                stats.add_progress(0, 1)
                break
            
            if stats:
                stats.add_progress(len(chunk.encode('utf-8')), 0)
            
            # Combine with leftover from previous chunk
            buffer += chunk
            
            # Find last comma to avoid splitting a number
            last_comma = buffer.rfind(',')
            if last_comma == -1:
                continue  # No complete number yet
            
            # Process complete numbers
            complete_part = buffer[:last_comma]
            buffer = buffer[last_comma + 1:]
            
            for num_str in complete_part.split(','):
                num_str = num_str.strip()
                if num_str:
                    try:
                        val = int(num_str)
                        if val > max_val:
                            max_val = val
                        if stats:
                            stats.add_progress(0, 1)
                    except ValueError:
                        pass  # Skip invalid numbers
    
    return int(max_val) if max_val != float('-inf') else 0


def find_max_in_range(filepath: str, start: int, end: int, 
                      chunk_size: int, stats: Optional[Stats] = None) -> int:
    """Find maximum in a byte range of the file."""
    max_val = float('-inf')
    
    with open(filepath, 'r') as f:
        f.seek(start)
        
        # Skip partial number at start (unless at beginning)
        if start > 0:
            # Read until we find a comma
            while True:
                char = f.read(1)
                if not char or char == ',':
                    break
        
        remaining = end - f.tell()
        buffer = ""
        
        while remaining > 0:
            to_read = min(chunk_size, remaining)
            chunk = f.read(to_read)
            if not chunk:
                break
            
            remaining -= len(chunk)
            
            if stats:
                stats.add_progress(len(chunk.encode('utf-8')), 0)
            
            buffer += chunk
            
            # Find last comma
            last_comma = buffer.rfind(',')
            if last_comma == -1:
                continue
            
            complete_part = buffer[:last_comma]
            buffer = buffer[last_comma + 1:]
            
            for num_str in complete_part.split(','):
                num_str = num_str.strip()
                if num_str:
                    try:
                        val = int(num_str)
                        if val > max_val:
                            max_val = val
                        if stats:
                            stats.add_progress(0, 1)
                    except ValueError:
                        pass
        
        # Process remaining buffer
        if buffer:
            for num_str in buffer.split(','):
                num_str = num_str.strip()
                if num_str:
                    try:
                        val = int(num_str)
                        if val > max_val:
                            max_val = val
                        if stats:
                            stats.add_progress(0, 1)
                    except ValueError:
                        pass
    
    return int(max_val) if max_val != float('-inf') else 0


def find_max_multi_thread(filepath: str, num_threads: int, 
                          chunk_size: int = 64 * 1024 * 1024,
                          stats: Optional[Stats] = None) -> int:
    """Find maximum using multiple threads."""
    file_size = os.path.getsize(filepath)
    segment_size = file_size // num_threads
    
    segments = []
    for i in range(num_threads):
        start = i * segment_size
        end = file_size if i == num_threads - 1 else (i + 1) * segment_size
        segments.append((start, end))
    
    max_val = float('-inf')
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(find_max_in_range, filepath, start, end, chunk_size, stats)
            for start, end in segments
        ]
        
        for future in as_completed(futures):
            result = future.result()
            if result > max_val:
                max_val = result
    
    return int(max_val)


def print_tabulated_results(results: dict):
    """Print results in tabulated format."""
    print(f"\n{'='*80}")
    print(f"{'BENCHMARK RESULTS SUMMARY':^80}")
    print(f"{'='*80}")
    
    print(f"\n{'METRIC':<40} {'VALUE':>35}")
    print(f"{'-'*40} {'-'*35}")
    
    file_size_gb = results['file_size'] / (1024**3)
    print(f"{'File Size':<40} {file_size_gb:>32.2f} GB")
    print(f"{'Threads Used':<40} {results['num_threads']:>35}")
    print(f"{'Maximum Value Found':<40} {results['max_value']:>35}")
    print(f"{'Total Execution Time':<40} {results['elapsed_time']:>32.2f} sec")
    
    print(f"\n{'--- DISK THROUGHPUT ---':<40}")
    print(f"{'Average Throughput':<40} {results['throughput']['avg_throughput']:>30.2f} MB/s")
    print(f"{'Peak Throughput':<40} {results['throughput']['peak_throughput']:>30.2f} MB/s")
    
    print(f"\n{'--- I/O TIMING ---':<40}")
    io = results['io_stats']
    print(f"{'Wall Clock Time':<40} {io['wall_time']:>32.2f} sec")
    print(f"{'CPU Time (user+sys)':<40} {io['cpu_time']:>32.2f} sec")
    print(f"{'I/O Wait Time (estimated)':<40} {io['io_wait_time']:>32.2f} sec")
    print(f"{'I/O Wait Percentage':<40} {io['io_wait_pct']:>32.1f} %")
    
    print(f"\n{'--- CPU CORES ---':<40}")
    cpu = results['cpu']
    print(f"{'Total Logical Cores':<40} {cpu['total_cores']:>35}")
    print(f"{'Cores Actively Used':<40} {cpu['cores_used']:>35}")
    
    if cpu.get('per_core_times'):
        print(f"\n{'--- PER-CORE CPU TIME ---'}")
        print(f"{'Core':<8} {'On-CPU Time':>14} {'Idle Time':>14} {'On-CPU %':>12} {'Util %':>10}")
        print(f"{'-'*8} {'-'*14} {'-'*14} {'-'*12} {'-'*10}")
        
        for i, core_time in enumerate(cpu['per_core_times']):
            util_pct = cpu['per_core_avg'][i] if i < len(cpu['per_core_avg']) else 0
            marker = "*" if i in cpu.get('consistently_active_cores', []) else " "
            print(f"Core {i:<2}{marker} {core_time['on_cpu_time']:>13.2f}s "
                  f"{core_time['idle_time']:>13.2f}s {core_time['on_cpu_pct']:>11.1f}% {util_pct:>9.1f}%")
    
    print(f"\n{'='*80}\n")


def run_benchmark(filepath: str, num_threads: int = 1, chunk_size_mb: int = 64) -> dict:
    """Run the benchmark with full monitoring."""
    chunk_size = chunk_size_mb * 1024 * 1024
    file_size = os.path.getsize(filepath)
    
    print(f"\n{'='*60}")
    print(f"Text File Max Finder Benchmark (Python)")
    print(f"{'='*60}")
    print(f"File: {filepath}")
    print(f"File Size: {file_size / (1024**3):.2f} GB")
    print(f"Mode: {'Single-threaded' if num_threads == 1 else f'Multi-threaded ({num_threads} threads)'}")
    print(f"Format: Comma-separated integers (text)")
    print(f"{'='*60}\n")
    
    stats = Stats(start_time=time.time())
    cpu_monitor = CPUMonitor(interval=0.5)
    throughput_monitor = ThroughputMonitor(stats, interval=1.0)
    io_monitor = IOWaitMonitor()
    
    cpu_monitor.start()
    throughput_monitor.start()
    io_monitor.start()
    
    start_time = time.time()
    
    try:
        if num_threads == 1:
            max_val = find_max_single_thread(filepath, chunk_size, stats)
        else:
            max_val = find_max_multi_thread(filepath, num_threads, chunk_size, stats)
    finally:
        throughput_monitor.stop()
        cpu_monitor.stop()
    
    elapsed = time.time() - start_time
    
    return {
        'max_value': max_val,
        'elapsed_time': elapsed,
        'throughput': throughput_monitor.get_summary(),
        'cpu': cpu_monitor.get_summary(),
        'io_stats': io_monitor.get_stats(),
        'file_size': file_size,
        'num_threads': num_threads,
        'numbers_read': stats.numbers_read
    }


def main():
    parser = argparse.ArgumentParser(
        description='Find maximum integer in a comma-separated text file',
        epilog="""
Examples:
  python max_finder_text.py data.txt --threads 1
  python max_finder_text.py data.txt --threads 10
  python max_finder_text.py data.txt --threads 0  # all cores
        """
    )
    
    parser.add_argument('filepath', help='Path to text file')
    parser.add_argument('-t', '--threads', type=int, default=1,
                        help='Number of threads (0 = all cores)')
    parser.add_argument('-c', '--chunk-size', type=int, default=64,
                        help='Chunk size in MB')
    parser.add_argument('--csv', type=str, help='Append results to CSV')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.filepath):
        print(f"Error: File not found: {args.filepath}", file=sys.stderr)
        sys.exit(1)
    
    num_threads = args.threads
    if num_threads == 0:
        num_threads = psutil.cpu_count(logical=True)
        print(f"Using all {num_threads} logical cores")
    
    try:
        results = run_benchmark(args.filepath, num_threads, args.chunk_size)
        print_tabulated_results(results)
        
        print(f"Summary: Found max={results['max_value']} in {results['elapsed_time']:.2f}s")
        
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted")
        sys.exit(130)


if __name__ == '__main__':
    main()
