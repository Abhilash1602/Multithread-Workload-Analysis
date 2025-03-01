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
            current_metrics['execution_time'] = 0  # Initialize execution time
            continue
        
        # Extract total program execution time
        total_time_match = re.search(r'Total program execution time: (\d+\.\d+) seconds', line)
        if total_time_match:
            current_metrics['total_execution_time'] = float(total_time_match.group(1))
            current_metrics['execution_time'] = float(total_time_match.group(1))  # Set execution time
            continue

        # Extract matrix multiplication time
        exec_time_match = re.search(r'Matrix multiplication completed in (\d+\.\d+) seconds', line)
        if exec_time_match:
            current_metrics['matrix_mult_time'] = float(exec_time_match.group(1))
            continue

        # Extract additional performance metrics
        metrics_patterns = {
            'context-switches': r'^\s*(\d+,?\d*)\s+context-switches',
            'cpu-migrations': r'^\s*(\d+,?\d*)\s+cpu-migrations',
            'cache-references': r'^\s*(\d+,?\d*)\s+cache-references',
            'cache-misses': r'^\s*(\d+,?\d*)\s+cache-misses',
            'L1-dcache-loads': r'^\s*(\d+,?\d*)\s+L1-dcache-loads',
            'L1-dcache-load-misses': r'^\s*(\d+,?\d*)\s+L1-dcache-load-misses',
            'branch-misses': r'^\s*(\d+,?\d*)\s+branch-misses',
            'cpu-usage': r'^\s*(\d+\.\d+)\s+CPUs utilized',
            'mem-loads': r'^\s*(\d+,?\d*)\s+mem-loads',
            'mem-stores': r'^\s*(\d+,?\d*)\s+mem-stores',
            'task-clock': r'^\s*(\d+\.\d+)\s+msec task-clock',
            'cycles': r'^\s*(\d+,?\d*)\s+cycles',
            'instructions': r'^\s*(\d+,?\d*)\s+instructions'
        }
        
        for metric, pattern in metrics_patterns.items():
            match = re.search(pattern, line)
            if match:
                value = float(match.group(1).replace(',', ''))
                current_metrics[metric] = value
    
    if current_metrics:
        data.append(current_metrics)
    
    return pd.DataFrame(data)

def create_execution_time_plot(df_concurrent, df_parallel):
    """Create plot specifically for execution times comparing concurrent and parallel implementations."""
    plt.figure(figsize=(12, 6))
    plt.style.use('dark_background')
    
    # Plot concurrent implementation
    plt.semilogx(df_concurrent['threads'], df_concurrent['total_execution_time'],
             marker='o', label='Concurrent - Total Time',
             color='yellow', linewidth=2.5, markersize=8)
    plt.semilogx(df_concurrent['threads'], df_concurrent['matrix_mult_time'],
             marker='s', label='Concurrent - Matrix Mult Time',
             color='cyan', linewidth=2.5, markersize=8)
    
    # Plot parallel implementation
    plt.semilogx(df_parallel['threads'], df_parallel['total_execution_time'],
             marker='o', label='Parallel - Total Time',
             color='red', linewidth=2.5, markersize=8)
    plt.semilogx(df_parallel['threads'], df_parallel['matrix_mult_time'],
             marker='s', label='Parallel - Matrix Mult Time',
             color='magenta', linewidth=2.5, markersize=8)
    
    plt.xlabel('Number of Threads (log scale)')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Concurrent vs Parallel Implementation Performance', pad=20, fontsize=12)
    
    thread_list = [1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 50, 64, 100, 128, 150, 200]
    plt.xticks(thread_list, thread_list, rotation=45)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.legend(fontsize=10, framealpha=0.8, title='Performance Metrics')
    plt.tight_layout()
    
    plt.savefig('execution_times_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_plot(df_concurrent, df_parallel, metric_name, ylabel, title, filename):
    """Create plot for performance metrics comparing concurrent and parallel implementations."""
    plt.figure(figsize=(12, 6))
    plt.style.use('dark_background')
    
    # Plot concurrent implementation
    plt.semilogx(df_concurrent['threads'], df_concurrent[metric_name],
             marker='o', label='Concurrent Implementation',
             color='yellow', linewidth=2.5, markersize=8)
    
    # Plot parallel implementation
    plt.semilogx(df_parallel['threads'], df_parallel[metric_name],
             marker='s', label='Parallel Implementation',
             color='cyan', linewidth=2.5, markersize=8)
    
    plt.xlabel('Number of Threads (log scale)')
    plt.ylabel(ylabel)
    plt.title(f'{title} - Concurrent vs Parallel', pad=20, fontsize=12)
    
    thread_list = [1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 50, 64, 100, 128, 150, 200]
    plt.xticks(thread_list, thread_list, rotation=45)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Add trend lines
    for df, label, color in [(df_concurrent, 'Concurrent Trend', 'white'), 
                            (df_parallel, 'Parallel Trend', 'red')]:
        z = np.polyfit(np.log10(df['threads']), df[metric_name], 1)
        p = np.poly1d(z)
        plt.plot(df['threads'], p(np.log10(df['threads'])), 
                "--", color=color, alpha=0.5, label=label)
    
    plt.legend(fontsize=10, framealpha=0.8)
    plt.tight_layout()
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_performance():
    """Analyze and plot various performance metrics for both implementations."""
    # Load data for both implementations
    df_concurrent = parse_perf_file('output_c_c.txt')
    df_parallel = parse_perf_file('output_c_p.txt')
    
    # Create execution time comparison plot
    create_execution_time_plot(df_concurrent, df_parallel)
    
    # Calculate derived metrics for both dataframes
    for df in [df_concurrent, df_parallel]:
        if 'mem-loads' in df.columns and 'mem-stores' in df.columns and 'task-clock' in df.columns:
            df['memory_bandwidth'] = (df['mem-loads'] + df['mem-stores']) / (df['task-clock'] / 1000)
        
        if 'instructions' in df.columns and 'cycles' in df.columns:
            df['ipc'] = df['instructions'] / df['cycles']
    
    # Define metrics to plot
    metrics_to_plot = [
        ('total_execution_time', 'Time (seconds)', 'Total Execution Time', 'execution_time.png'),
        ('context-switches', 'Switches/sec', 'Context Switch Rate', 'context_switches.png'),
        ('cpu-migrations', 'Migrations/sec', 'CPU Migration Rate', 'cpu_migrations.png'),
        ('cache-references', 'References/sec', 'Cache Reference Rate', 'cache_references.png'),
        ('cache-misses', 'Miss Rate (%)', 'Cache Miss Rate', 'cache_misses.png'),
        ('L1-dcache-load-misses', 'Miss Rate (%)', 'L1 Cache Miss Rate', 'l1_cache_misses.png'),
        ('branch-misses', 'Misses/sec', 'Branch Miss Rate', 'branch_misses.png'),
        ('cpu-usage', 'CPUs', 'CPU Utilization', 'cpu_usage.png'),
        ('memory_bandwidth', 'Operations/sec', 'Memory Bandwidth', 'memory_bandwidth.png'),
        ('ipc', 'Instructions/Cycle', 'Instructions per Cycle', 'ipc.png')
    ]
    
    # Create plots for each metric
    for metric, ylabel, title, filename in metrics_to_plot:
        try:
            create_performance_plot(df_concurrent, df_parallel, metric, ylabel, title, filename)
        except KeyError as e:
            print(f"Skipping metric {metric}: not found in data")

def generate_analysis_report(df):
    """Generate a detailed performance analysis report."""
    with open('performance_analysis_report.txt', 'w') as f:
        f.write("Performance Analysis Report\n")
        f.write("==========================\n\n")
        
        # Overall Performance
        f.write("Overall Performance:\n")
        f.write("-----------------\n")
        min_time_idx = df['total_execution_time'].idxmin()
        optimal_threads = df.loc[min_time_idx, 'threads']
        f.write(f"Optimal thread count: {optimal_threads}\n")
        f.write(f"Best execution time: {df.loc[min_time_idx, 'total_execution_time']:.3f} seconds\n")
        speedup = df.iloc[0]['total_execution_time'] / df.loc[min_time_idx, 'total_execution_time']
        f.write(f"Speedup vs single thread: {speedup:.2f}x\n\n")
        
        # Thread Behavior Analysis
        f.write("Thread Behavior Analysis:\n")
        f.write("----------------------\n")
        if 'context-switches' in df.columns:
            switches_per_thread = df['context-switches'] / df['threads']
            f.write("Context switches per thread:\n")
            f.write(f"  Minimum: {switches_per_thread.min():.2f}\n")
            f.write(f"  Maximum: {switches_per_thread.max():.2f}\n")
            f.write(f"  Average: {switches_per_thread.mean():.2f}\n\n")
            
            # High context switch detection
            high_switches = df[switches_per_thread > switches_per_thread.mean() + switches_per_thread.std()]
            if not high_switches.empty:
                f.write("High context switch overhead detected at thread counts: ")
                f.write(", ".join(map(str, high_switches['threads'].tolist())))
                f.write("\n\n")
        
        # Memory System Analysis
        f.write("Memory System Performance:\n")
        f.write("------------------------\n")
        if 'L1-dcache-load-misses' in df.columns and 'L1-dcache-loads' in df.columns:
            l1_miss_rates = (df['L1-dcache-load-misses'] / df['L1-dcache-loads'] * 100)
            f.write("L1 Cache Performance:\n")
            f.write(f"  Average miss rate: {l1_miss_rates.mean():.2f}%\n")
            f.write(f"  Minimum miss rate: {l1_miss_rates.min():.2f}%\n")
            f.write(f"  Maximum miss rate: {l1_miss_rates.max():.2f}%\n")
            
            # Identify problematic thread counts for L1 cache
            high_l1_misses = df[l1_miss_rates > l1_miss_rates.mean() + l1_miss_rates.std()]
            if not high_l1_misses.empty:
                f.write("  High miss rates at thread counts: ")
                f.write(", ".join(map(str, high_l1_misses['threads'].tolist())))
                f.write("\n")
        
        if 'cache-misses' in df.columns and 'cache-references' in df.columns:
            cache_miss_rates = (df['cache-misses'] / df['cache-references'] * 100)
            f.write("\nLLC Cache Performance:\n")
            f.write(f"  Average miss rate: {cache_miss_rates.mean():.2f}%\n")
            f.write(f"  Minimum miss rate: {cache_miss_rates.min():.2f}%\n")
            f.write(f"  Maximum miss rate: {cache_miss_rates.max():.2f}%\n")
            
            # Identify problematic thread counts for LLC
            high_cache_misses = df[cache_miss_rates > cache_miss_rates.mean() + cache_miss_rates.std()]
            if not high_cache_misses.empty:
                f.write("  High miss rates at thread counts: ")
                f.write(", ".join(map(str, high_cache_misses['threads'].tolist())))
                f.write("\n")
        
        # Producer-Consumer Specific Analysis
        f.write("\nProducer-Consumer Thread Pattern Analysis:\n")
        f.write("---------------------------------------\n")
        # Analyze scaling efficiency
        thread_efficiency = (df.iloc[0]['total_execution_time'] / df['total_execution_time']) / df['threads']
        f.write(f"Best scaling efficiency at {df.loc[thread_efficiency.idxmax(), 'threads']} threads\n")
        f.write(f"Worst scaling efficiency at {df.loc[thread_efficiency.idxmin(), 'threads']} threads\n\n")
        
        # Performance Bottlenecks
        f.write("Performance Bottlenecks:\n")
        f.write("----------------------\n")
        bottleneck_threads = set()
        
        # Context switch bottlenecks
        if 'context-switches' in df.columns:
            high_cs = df[switches_per_thread > switches_per_thread.mean() + switches_per_thread.std()]['threads']
            bottleneck_threads.update(high_cs)
        
        # Cache bottlenecks
        if 'cache-misses' in df.columns and 'cache-references' in df.columns:
            high_cache = df[cache_miss_rates > cache_miss_rates.mean() + cache_miss_rates.std()]['threads']
            bottleneck_threads.update(high_cache)
        
        if bottleneck_threads:
            f.write("Potential bottlenecks detected at thread counts: ")
            f.write(", ".join(map(str, sorted(bottleneck_threads))))
            f.write("\n")

def generate_comparative_analysis(df_concurrent, df_parallel):
    """Generate a comparative analysis report between concurrent and parallel implementations."""
    with open('comparative_analysis_report.txt', 'w') as f:
        f.write("Comparative Analysis Report: Concurrent vs Parallel Implementation\n")
        f.write("========================================================\n\n")
        
        # Overall Performance Comparison
        f.write("1. Overall Performance Comparison\n")
        f.write("--------------------------------\n")
        
        # Best execution times
        c_min_idx = df_concurrent['total_execution_time'].idxmin()
        p_min_idx = df_parallel['total_execution_time'].idxmin()
        
        f.write("\nBest Performance Points:\n")
        f.write(f"Concurrent: {df_concurrent.loc[c_min_idx, 'total_execution_time']:.3f} seconds with {df_concurrent.loc[c_min_idx, 'threads']} threads\n")
        f.write(f"Parallel  : {df_parallel.loc[p_min_idx, 'total_execution_time']:.3f} seconds with {df_parallel.loc[p_min_idx, 'threads']} threads\n")
        
        # Performance difference
        perf_diff = ((df_concurrent['total_execution_time'].min() - df_parallel['total_execution_time'].min()) 
                    / df_concurrent['total_execution_time'].min() * 100)
        f.write(f"\nPerformance Difference: {abs(perf_diff):.2f}% ")
        f.write("better with parallel implementation\n" if perf_diff > 0 else "better with concurrent implementation\n")
        
        # Scaling Analysis
        f.write("\n2. Scaling Analysis\n")
        f.write("------------------\n")
        
        for threads in [1, 4, 8, 16, 32, 64, 128]:
            if threads in df_concurrent['threads'].values and threads in df_parallel['threads'].values:
                c_time = df_concurrent[df_concurrent['threads'] == threads]['total_execution_time'].iloc[0]
                p_time = df_parallel[df_parallel['threads'] == threads]['total_execution_time'].iloc[0]
                f.write(f"\nThread count: {threads}")
                f.write(f"\nConcurrent: {c_time:.3f}s, Parallel: {p_time:.3f}s")
                f.write(f"\nDifference: {abs(c_time - p_time):.3f}s ({abs((c_time - p_time)/c_time * 100):.2f}%)")
        
        # Resource Utilization
        f.write("\n\n3. Resource Utilization\n")
        f.write("----------------------\n")
        
        # CPU Usage
        if 'cpu-usage' in df_concurrent.columns and 'cpu-usage' in df_parallel.columns:
            f.write("\nCPU Utilization:")
            f.write(f"\nConcurrent - Avg: {df_concurrent['cpu-usage'].mean():.2f}, Max: {df_concurrent['cpu-usage'].max():.2f}")
            f.write(f"\nParallel   - Avg: {df_parallel['cpu-usage'].mean():.2f}, Max: {df_parallel['cpu-usage'].max():.2f}")
        
        # Cache Performance
        if 'cache-misses' in df_concurrent.columns and 'cache-misses' in df_parallel.columns:
            f.write("\n\nCache Miss Rates:")
            c_miss_rate = (df_concurrent['cache-misses'] / df_concurrent['cache-references'] * 100).mean()
            p_miss_rate = (df_parallel['cache-misses'] / df_parallel['cache-references'] * 100).mean()
            f.write(f"\nConcurrent - Avg: {c_miss_rate:.2f}%")
            f.write(f"\nParallel   - Avg: {p_miss_rate:.2f}%")
        
        # Context Switches
        if 'context-switches' in df_concurrent.columns and 'context-switches' in df_parallel.columns:
            f.write("\n\nContext Switches per Thread:")
            c_switches = (df_concurrent['context-switches'] / df_concurrent['threads']).mean()
            p_switches = (df_parallel['context-switches'] / df_parallel['threads']).mean()
            f.write(f"\nConcurrent - Avg: {c_switches:.2f}")
            f.write(f"\nParallel   - Avg: {p_switches:.2f}")
        
        # Recommendations
        f.write("\n\n4. Recommendations\n")
        f.write("-----------------\n")
        
        # Based on thread count
        optimal_c_threads = df_concurrent.loc[c_min_idx, 'threads']
        optimal_p_threads = df_parallel.loc[p_min_idx, 'threads']
        f.write(f"\nOptimal thread counts:")
        f.write(f"\n- Concurrent: {optimal_c_threads}")
        f.write(f"\n- Parallel  : {optimal_p_threads}")
        
        # Overall recommendation
        f.write("\n\nOverall Recommendation:")
        if perf_diff > 10:  # More than 10% difference
            f.write("\nParallel implementation shows significantly better performance.")
        elif perf_diff < -10:
            f.write("\nConcurrent implementation shows significantly better performance.")
        else:
            f.write("\nBoth implementations show comparable performance.")

def main():
    """Main function to run all analyses."""
    df_concurrent = parse_perf_file('output_c_c.txt')
    df_parallel = parse_perf_file('output_c_p.txt')
    
    analyze_performance()
    generate_comparative_analysis(df_concurrent, df_parallel)
    print("Analysis complete. Check execution_time.png, performance_analysis_report.txt, and comparative_analysis_report.txt")

if __name__ == "__main__":
    main()
