import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def parse_perf_file(filename):
    """Parse perf stat output file and extract metrics."""
    data = []
    current_metrics = {}
    
    with open(filename, 'r') as f:
        content = f.readlines()
    
    for line in content:
        # Extract thread count
        thread_match = re.search(r'Running with (\d+) threads', line)
        if thread_match:
            if current_metrics:
                data.append(current_metrics.copy())
            current_metrics = {'threads': int(thread_match.group(1))}
            continue
        
        # Extract metrics with improved patterns
        patterns = {
            'cycles': r'(\d+,?\d*)\s+(?:cpu_(?:atom|core)/)?cycles/',  # Modified pattern
            'instructions': r'(\d+,?\d*)\s+(?:cpu_(?:atom|core)/)?instructions/',  # Modified pattern
            'cache-references': r'(\d+,?\d*)\s+(?:cpu_(?:atom|core)/)?cache-references/',  # Modified pattern
            'cache-misses': r'(\d+,?\d*)\s+(?:cpu_(?:atom|core)/)?cache-misses/',  # Modified pattern
            'branch-misses': r'(\d+,?\d*)\s+(?:cpu_(?:atom|core)/)?branch-misses/',  # Modified pattern
            'context-switches': r'(\d+,?\d*)\s+context-switches',
            'cpu-migrations': r'(\d+,?\d*)\s+cpu-migrations',
            'page-faults': r'(\d+,?\d*)\s+page-faults',
            'L1-dcache-loads': r'(\d+,?\d*)\s+(?:cpu_(?:atom|core)/)?L1-dcache-loads/',  # Modified pattern
            'L1-dcache-misses': r'(\d+,?\d*)\s+(?:cpu_(?:atom|core)/)?L1-dcache-load-misses/',  # Modified pattern
            'LLC-loads': r'(\d+,?\d*)\s+(?:cpu_(?:atom|core)/)?LLC-loads/',  # Modified pattern
            'LLC-misses': r'(\d+,?\d*)\s+(?:cpu_(?:atom|core)/)?LLC-load-misses/',  # Modified pattern
            'dTLB-loads': r'(\d+,?\d*)\s+(?:cpu_(?:atom|core)/)?dTLB-loads/',  # Modified pattern
            'dTLB-misses': r'(\d+,?\d*)\s+(?:cpu_(?:atom|core)/)?dTLB-load-misses/',  # Modified pattern
            'task-clock': r'(\d+\.\d+)\s+msec task-clock',
            'cpu-utilization': r'(\d+\.\d+)\s+CPUs utilized',
            'mem-loads': r'(\d+,?\d*)\s+(?:cpu_(?:atom|core)/)?mem-loads/',  # Modified pattern
            'mem-stores': r'(\d+,?\d*)\s+(?:cpu_(?:atom|core)/)?mem-stores/',  # Modified pattern
            'execution_time': r'(\d+\.\d+)\s+seconds time elapsed'
        }
        
        for metric, pattern in patterns.items():
            match = re.search(pattern, line)
            if match:
                try:
                    value = float(match.group(1).replace(',', ''))
                    # Sum up values if metric already exists (for cpu_atom and cpu_core)
                    current_metrics[metric] = current_metrics.get(metric, 0) + value
                except (ValueError, IndexError):
                    continue
    
    if current_metrics:
        data.append(current_metrics)
    
    return pd.DataFrame(data)

def create_performance_plot(programs, metric_name, ylabel, title):
    """Create plot for performance metrics with integer x-axis."""
    plt.figure(figsize=(12, 6))
    colors = {
        'CPU Bound': '#1f77b4',
        'Memory Bound': '#ff7f0e',
        'I/O Bound': '#2ca02c',
        'Mixed Workload': '#d62728'
    }
    
    for program_name, df in programs.items():
        plt.plot(df['threads'], df[metric_name],
                marker='o', label=program_name,
                color=colors[program_name],
                linewidth=2, markersize=8)
    
    plt.xlabel('Number of Threads')
    plt.ylabel(ylabel)
    plt.title(title)
    
    # Use integer values for x-axis ticks
    unique_threads = sorted(set([t for df in programs.values() for t in df['threads']]))
    plt.xticks(unique_threads)
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    metric_filename = metric_name.replace('-', '_').lower()
    plt.savefig(f'metric_{metric_filename}.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_performance():
    """Analyze and plot various performance metrics."""
    # Load data
    programs = {
        'CPU Bound': parse_perf_file('output1.txt'),
        'Memory Bound': parse_perf_file('output3.txt'),
        'I/O Bound': parse_perf_file('output2.txt'),
        'Mixed Workload': parse_perf_file('output4.txt')
    }
    
    # Updated metrics list with only available metrics
    metrics_to_plot = [
        ('cycles', 'CPU Cycles', 'CPU Cycles'),
        ('instructions', 'Instructions', 'Instructions'),
        ('cache-references', 'Cache References', 'Cache References'),
        ('cache-misses', 'Cache Misses', 'Cache Misses'),
        ('branch-misses', 'Branch Misses', 'Branch Misses'),
        ('context-switches', 'Context Switches', 'Context Switches'),
        ('cpu-migrations', 'CPU Migrations', 'CPU Migrations'),
        ('page-faults', 'Page Faults', 'Page Faults'),
        ('L1-dcache-loads', 'L1 D-Cache Loads', 'L1 Cache Loads'),
        ('LLC-loads', 'Last Level Cache Loads', 'LLC Loads'),
        ('LLC-misses', 'Last Level Cache Misses', 'LLC Misses'),
        ('cpu-utilization', 'CPU Utilization', 'CPUs Utilized'),
        ('task-clock', 'Task Clock', 'Task Clock (ms)'),
        ('execution_time', 'Execution Time', 'Time (seconds)')
    ]
    
    # Create plots for each metric
    for metric_name, title, ylabel in metrics_to_plot:
        try:
            create_performance_plot(programs, metric_name, ylabel, title)
        except KeyError as e:
            print(f"Skipping metric {metric_name}: not found in data")
    
    # Generate summary report
    generate_analysis_report(programs)

def generate_analysis_report(programs):
    """Generate detailed performance analysis report."""
    with open('performance_analysis3.txt', 'w') as f:
        f.write("Performance Analysis Report\n")
        f.write("=========================\n\n")
        
        for name, df in programs.items():
            f.write(f"\n{name} Analysis:\n")
            f.write("=" * (len(name) + 10) + "\n")
            
            # CPU efficiency analysis
            f.write("\nCPU Efficiency:\n")
            f.write(f"- Average CPU utilization: {df['cpu-utilization'].mean():.2f} CPUs\n")
            f.write(f"- Instructions per cycle: {(df['instructions'] / df['cycles']).mean():.2f}\n")
            
            # Memory analysis
            f.write("\nMemory Performance:\n")
            f.write(f"- Average cache miss rate: {(df['cache-misses'] / df['cache-references'] * 100).mean():.2f}%\n")
            f.write(f"- Average L1 cache miss rate: {(df['L1-dcache-misses'] / df['L1-dcache-loads'] * 100).mean():.2f}%\n")
            
            # Thread scaling analysis
            f.write("\nThread Scaling:\n")
            optimal_threads = df.loc[df['execution_time'].idxmin(), 'threads']
            f.write(f"- Optimal thread count: {optimal_threads}\n")
            f.write(f"- Speedup at optimal threads: {df['execution_time'].iloc[0] / df['execution_time'].min():.2f}x\n")
            
            # System overhead
            f.write("\nSystem Overhead:\n")
            f.write(f"- Average context switches per thread: {df['context-switches'].mean() / df['threads'].mean():.2f}\n")
            f.write(f"- Average CPU migrations per thread: {df['cpu-migrations'].mean() / df['threads'].mean():.2f}\n")
            
            f.write("\n" + "-"*50 + "\n")

def main():
    analyze_performance()
    print("Analysis complete. Check the generated plots and performance_analysis3.txt")

if __name__ == "__main__":
    main()
