import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def parse_perf_file(filename):
    """Parse perf stat output file and extract metrics."""
    data = []
    current_metrics = {}
    thread_count = None
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # Split the file by section separators
    sections = content.split('---------------------------------------------------')
    
    for section in sections:
        if not section.strip():
            continue
            
        # Extract thread count
        thread_match = re.search(r'Running with (\d+) threads', section)
        if thread_match:
            thread_count = int(thread_match.group(1))
            current_metrics = {'threads': thread_count}
            
            # Extract execution time
            exec_time_match = re.search(r'Total program execution time: (\d+\.\d+) seconds', section)
            if exec_time_match:
                current_metrics['execution_time'] = float(exec_time_match.group(1))
            
            # Extract performance metrics
            patterns = {
                'cycles': r'(\d+,?\d*)\s+cycles',
                'instructions': r'(\d+,?\d*)\s+instructions',
                'cache-references': r'(\d+,?\d*)\s+cache-references',
                'cache-misses': r'(\d+,?\d*)\s+cache-misses',
                'branch-misses': r'(\d+,?\d*)\s+branch-misses',
                'context-switches': r'(\d+,?\d*)\s+context-switches',
                'cpu-migrations': r'(\d+,?\d*)\s+cpu-migrations',
                'page-faults': r'(\d+,?\d*)\s+page-faults',
                'L1-dcache-loads': r'(\d+,?\d*)\s+L1-dcache-loads',
                'L1-dcache-misses': r'(\d+,?\d*)\s+L1-dcache-load-misses',
                'LLC-loads': r'(\d+,?\d*)\s+LLC-loads',
                'LLC-misses': r'(\d+,?\d*)\s+LLC-load-misses',
                'dTLB-loads': r'(\d+,?\d*)\s+dTLB-loads',
                'dTLB-misses': r'(\d+,?\d*)\s+dTLB-load-misses',
                'task-clock': r'(\d+\.\d+)\s+msec task-clock',
                'cpu-utilization': r'(\d+\.\d+)\s+CPUs utilized',
                'mem-loads': r'(\d+,?\d*)\s+mem-loads',
                'mem-stores': r'(\d+,?\d*)\s+mem-stores',
                'seconds_elapsed': r'(\d+\.\d+)\s+seconds time elapsed'
            }
            
            for metric, pattern in patterns.items():
                match = re.search(pattern, section)
                if match:
                    value = float(match.group(1).replace(',', ''))
                    current_metrics[metric] = value
            
            if current_metrics:
                data.append(current_metrics.copy())
    
    return pd.DataFrame(data)

def create_performance_plot(df, metric_name, ylabel, title):
    """Create plot for performance metrics with logarithmic x-axis."""
    plt.figure(figsize=(12, 6))
    
    # Plot with yellow line on black grid
    plt.plot(df['threads'], df[metric_name],
            marker='o', color='yellow',
            linewidth=2, markersize=8)
    
    plt.xlabel('Number of Threads (log₂ scale)', color='white')
    plt.ylabel(ylabel, color='white')
    plt.title(title, color='white')
    
    # Use logarithmic scale for x-axis
    plt.xscale('log', base=2)
    
    # Get unique thread values for x-ticks with log scale notation
    unique_threads = sorted(set(df['threads']))
    plt.xticks(unique_threads, [f"{t} (2^{np.log2(t):.0f})" if np.log2(t).is_integer() else str(t) for t in unique_threads], 
              color='white', rotation=45, ha='right')
    plt.yticks(color='white')
    
    # Adjust bottom margin to accommodate rotated labels
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Set visible grid with alpha
    plt.grid(True, alpha=0.4, color='gray', linestyle='--', which='both')
    plt.gca().set_facecolor('black')
    plt.gcf().set_facecolor('black')
    
    # Add borders for better visibility
    plt.gca().spines['bottom'].set_color('white')
    plt.gca().spines['top'].set_color('white') 
    plt.gca().spines['right'].set_color('white')
    plt.gca().spines['left'].set_color('white')
    
    # Add minor grid lines
    plt.minorticks_on()
    
    # Save plot
    metric_filename = metric_name.replace('-', '_').lower()
    plt.savefig(f'metric_{metric_filename}.png', dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()

def analyze_performance():
    """Analyze and plot various performance metrics."""
    # Load data from single file
    df = parse_perf_file('output_c.txt')
    
    # Print summary to verify data parsing
    print(f"Parsed data for {len(df)} thread configurations")
    print(f"Thread counts: {sorted(df['threads'].unique())}")
    
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
        ('L1-dcache-misses', 'L1 D-Cache Misses', 'L1 Cache Load Misses'),
        ('LLC-loads', 'Last Level Cache Loads', 'LLC Loads'),
        ('LLC-misses', 'Last Level Cache Misses', 'LLC Misses'),
        ('task-clock', 'Task Clock', 'Task Clock (ms)'),
        ('cpu-utilization', 'CPU Utilization', 'CPUs Utilized'),
        ('execution_time', 'Execution Time', 'Time (seconds)'),
        ('seconds_elapsed', 'Time Elapsed', 'Seconds Elapsed')
    ]
    
    # Create plots for each metric
    for metric_name, title, ylabel in metrics_to_plot:
        try:
            create_performance_plot(df, metric_name, ylabel, title)
            print(f"Created plot for {metric_name}")
        except KeyError as e:
            print(f"Skipping metric {metric_name}: not found in data")
    
    # Generate summary report
    generate_analysis_report(df)

def generate_analysis_report(df):
    """Generate detailed performance analysis report."""
    with open('performance_analysis.txt', 'w') as f:
        f.write("Performance Analysis Report\n")
        f.write("=========================\n\n")
        
        f.write("Thread Scaling Performance\n")
        f.write("-----------------------\n")
        f.write(f"Threads tested: {', '.join(map(str, sorted(df['threads'].unique())))}\n\n")
        
        # Execution time analysis
        f.write("Execution Time Analysis:\n")
        min_time_idx = df['execution_time'].idxmin()
        optimal_threads = df.loc[min_time_idx, 'threads']
        min_time = df.loc[min_time_idx, 'execution_time']
        base_time = df.loc[df['threads'] == 1, 'execution_time'].values[0] if 1 in df['threads'].values else df['execution_time'].max()
        
        f.write(f"- Best execution time: {min_time:.3f} seconds with {optimal_threads} threads\n")
        f.write(f"- Speedup over single thread: {base_time/min_time:.2f}x\n\n")
        
        # CPU efficiency analysis
        f.write("CPU Efficiency:\n")
        f.write(f"- Average IPC (Instructions per Cycle): {(df['instructions'] / df['cycles']).mean():.3f}\n")
        f.write(f"- Best IPC: {(df['instructions'] / df['cycles']).max():.3f} with {df.loc[(df['instructions'] / df['cycles']).idxmax(), 'threads']} threads\n")
        f.write(f"- Average CPU utilization: {df['cpu-utilization'].mean():.2f} CPUs\n\n")
        
        # Memory analysis
        f.write("Memory Performance:\n")
        f.write(f"- Average cache miss rate: {(df['cache-misses'] / df['cache-references'] * 100).mean():.2f}%\n")
        f.write(f"- Best cache miss rate: {(df['cache-misses'] / df['cache-references'] * 100).min():.2f}% with {df.loc[(df['cache-misses'] / df['cache-references']).idxmin(), 'threads']} threads\n")
        f.write(f"- Average L1 cache miss rate: {(df['L1-dcache-misses'] / df['L1-dcache-loads'] * 100).mean():.2f}%\n")
        f.write(f"- Average LLC cache miss rate: {(df['LLC-misses'] / df['LLC-loads'] * 100).mean():.2f}%\n\n")
        
        # Threading overhead
        f.write("Threading Overhead:\n")
        f.write(f"- Context switches per thread: {df['context-switches'].mean() / df['threads'].mean():.2f}\n")
        f.write(f"- CPU migrations per thread: {df['cpu-migrations'].mean() / df['threads'].mean():.2f}\n\n")
        
        # Efficiency scaling for all threads
        f.write("Efficiency Scaling (All Threads):\n")
        f.write("--------------------------------\n")
        f.write("Threads | Exec Time | Speedup | Efficiency | CPU Util\n")
        
        thread_counts = sorted(df['threads'].unique())
        base_time = df.loc[df['threads'] == min(thread_counts), 'execution_time'].values[0]
        
        for thread in thread_counts:
            thread_data = df[df['threads'] == thread].iloc[0]
            speedup = base_time / thread_data['execution_time']
            efficiency = speedup / thread
            f.write(f"{thread:7d} | {thread_data['execution_time']:9.3f} | {speedup:7.2f}x | {efficiency:9.2f} | {thread_data['cpu-utilization']:8.2f}\n")
        
        f.write("\n")
        
        # Create comparison table for powers of 2 thread counts
        f.write("Thread Scaling Comparison (Powers of 2):\n")
        f.write("---------------------------------------\n")
        
        # Filter for powers of 2 (2, 4, 8, 16, etc.)
        powers_of_2 = [t for t in thread_counts if (t & (t-1) == 0) and t > 0]
        
        # Define metrics to compare
        metrics_to_compare = [
            ('execution_time', 'Execution Time (s)'),
            ('instructions', 'Instructions (M)'),
            ('cycles', 'CPU Cycles (M)'),
            ('cache-misses', 'Cache Misses (K)'),
            ('context-switches', 'Context Switches'),
            ('cpu-utilization', 'CPU Utilization'),
            ('L1-dcache-misses', 'L1 Cache Misses (K)'),
            ('LLC-misses', 'LLC Misses (K)')
        ]
        
        # Create header for table
        f.write(f"{'Metric':<25} | " + " | ".join(f"{t:>10}" for t in powers_of_2) + "\n")
        f.write("-" * 25 + "+" + "+".join(["-" * 12] * len(powers_of_2)) + "\n")
        
        # Fill table with data
        for metric_key, metric_name in metrics_to_compare:
            if metric_key in df.columns:
                f.write(f"{metric_name:<25} | ")
                
                for thread in powers_of_2:
                    if thread in df['threads'].values:
                        value = df.loc[df['threads'] == thread, metric_key].values[0]
                        
                        # Format large numbers for better readability
                        if 'Instructions' in metric_name or 'Cycles' in metric_name:
                            value = value / 1_000_000  # Convert to millions
                        elif 'Misses' in metric_name and not 'LLC' in metric_name:
                            value = value / 1_000  # Convert to thousands
                        
                        f.write(f"{value:>10.2f} | ")
                    else:
                        f.write(f"{' '*10} | ")
                f.write("\n")
        
        f.write("\n")
        
        # Add speedup and efficiency data
        if len(powers_of_2) > 0 and powers_of_2[0] in df['threads'].values:
            base_thread = powers_of_2[0]
            base_time = df.loc[df['threads'] == base_thread, 'execution_time'].values[0]
            
            f.write(f"{'Speedup':<25} | ")
            for thread in powers_of_2:
                if thread in df['threads'].values:
                    thread_time = df.loc[df['threads'] == thread, 'execution_time'].values[0]
                    speedup = base_time / thread_time
                    f.write(f"{speedup:>10.2f}x | ")
                else:
                    f.write(f"{' '*10} | ")
            f.write("\n")
            
            f.write(f"{'Parallel Efficiency':<25} | ")
            for thread in powers_of_2:
                if thread in df['threads'].values:
                    thread_time = df.loc[df['threads'] == thread, 'execution_time'].values[0]
                    speedup = base_time / thread_time
                    efficiency = speedup / (thread / base_thread)
                    f.write(f"{efficiency:>10.2f} | ")
                else:
                    f.write(f"{' '*10} | ")
            f.write("\n")

def main():
    analyze_performance()
    print("Analysis complete. Check the generated plots and performance_analysis.txt")

if __name__ == "__main__":
    main()
