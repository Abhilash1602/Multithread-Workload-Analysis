import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

def compile_programs():
    """Compile all C programs"""
    # Get current directory
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    
    c_files = ['memory_bound.c', 'io_bound.c', 'cpu_bound.c', 'mixed_workload.c']
    
    # Verify source files exist
    missing_files = [f for f in c_files if not os.path.exists(f)]
    if missing_files:
        print("Error: The following source files are missing:")
        for f in missing_files:
            print(f"  - {f}")
        return False
    
    print("\nCompiling C programs...")
    for c_file in tqdm(c_files, desc="Compiling"):
        try:
            executable = os.path.join(current_dir, c_file.replace('.c', ''))
            compile_cmd = ['gcc', '-o', executable, c_file, '-pthread']
            print(f"\nRunning: {' '.join(compile_cmd)}")
            
            result = subprocess.run(compile_cmd, 
                                 check=True, 
                                 capture_output=True,
                                 text=True)
            
            if result.stderr:
                print(f"Compiler output for {c_file}:")
                print(result.stderr)
            
            # Make executable
            os.chmod(executable, 0o755)
            print(f"Successfully compiled {c_file} to {executable}")
            
        except subprocess.CalledProcessError as e:
            print(f"\nError compiling {c_file}:")
            print(f"Command failed: {' '.join(e.cmd)}")
            print("Compiler error:")
            print(e.stderr)
            return False
        except Exception as e:
            print(f"\nUnexpected error while compiling {c_file}:")
            print(str(e))
            return False
    return True

def run_perf_analysis(programs, num_threads):
    current_dir = os.getcwd()
    
    # First verify all executables exist
    for program in programs:
        program_path = os.path.join(current_dir, program)
        if not os.path.exists(program_path):
            print(f"Error: Executable not found at: {program_path}")
            return None
        if not os.access(program_path, os.X_OK):
            print(f"Error: No execute permission for: {program_path}")
            return None

    # Dictionary to store results
    results = {
        'program': [],
        'cycles': [],
        'instructions': [],
        'cache_references': [],
        'cache_misses': [],
        'task_clock': [],
        'context_switches': [],
        'page_faults': [],
        'cpu_migrations': [],
        'branch_misses': []
    }
    
    for program in tqdm(programs, desc="Analyzing programs"):
        print(f"\nAnalyzing {program} with {num_threads} threads...")
        program_path = os.path.join(current_dir, program)
        
        # Build the perf command
        cmd_list = [
            'perf', 'stat',
            '-e', 'cycles,instructions,cache-references,cache-misses,task-clock,context-switches,page-faults,cpu-migrations,branch-misses',
            program_path
        ]
        print(f"Running command: {' '.join(cmd_list)}")
        
        try:
            # Create a process with pipes for input/output
            process = subprocess.Popen(
                cmd_list,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Send the number of threads to the program's stdin
            stdout, stderr = process.communicate(input=f"{num_threads}\n")
            
            if process.returncode != 0:
                print(f"Program {program} failed with return code {process.returncode}")
                print("stdout:", stdout)
                print("stderr:", stderr)
                continue
                
            # Initialize metrics for this program
            results['program'].append(program)
            metrics = ['cycles', 'instructions', 'cache_references', 'cache_misses', 
                      'task_clock', 'context_switches', 'page_faults', 
                      'cpu_migrations', 'branch_misses']
            
            for metric in metrics:
                results[metric].append(0)  # Default value
            
            # Extract metrics from perf output (in stderr for perf stat)
            for line in stderr.split('\n'):
                for metric in metrics:
                    search_term = metric.replace('_', '-')
                    if search_term in line:
                        try:
                            value = float(line.split()[0].replace(',',''))
                            results[metric][-1] = value  # Update the last entry
                        except (ValueError, IndexError):
                            print(f"Warning: Could not parse value for {metric}")
                
        except Exception as e:
            print(f"Error running perf for {program}:")
            print(str(e))
            return None
            
    return pd.DataFrame(results)

def create_performance_plots(df):
    print("Creating performance plots...")
    metrics = ['cycles', 'instructions', 'cache_references', 'cache_misses', 'task_clock', 
              'context_switches', 'page_faults', 'cpu_migrations', 'branch_misses']
    
    # Define a color palette for each program
    colors = {
        'cpu_bound': '#FF9999',     # Red shade
        'memory_bound': '#66B2FF',  # Blue shade
        'io_bound': '#99FF99',      # Green shade
        'mixed_workload': '#FFCC99' # Orange shade
    }
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    for idx, metric in enumerate(tqdm(metrics, desc="Creating plots")):
        ax = axes[idx//3, idx%3]
        
        # Create barplot with custom colors
        sns.barplot(
            data=df, 
            x='program', 
            y=metric, 
            ax=ax,
            palette=colors,
            hue='program' if idx == 0 else None  # Only add hue for first plot
        )
        
        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        
        # Adjust legend only for first plot
        if idx == 0:
            ax.legend(title='Programs', bbox_to_anchor=(1.05, 1), loc='upper left')
        
    plt.tight_layout()
    plt.savefig('performance_comparison.png', bbox_inches='tight')
    print("Saved performance comparison plot as 'performance_comparison.png'")
    
    # Create cache miss rate plot with same color scheme
    plt.figure(figsize=(10, 6))
    df['cache_miss_rate'] = df['cache_misses'] / df['cache_references'] * 100
    
    sns.barplot(
        data=df, 
        x='program', 
        y='cache_miss_rate',
        palette=colors,
        hue='program'
    )
    
    plt.title('Cache Miss Rate Comparison')
    plt.xticks(rotation=45)
    plt.ylabel('Cache Miss Rate (%)')
    plt.legend(title='Programs', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('cache_miss_rate.png', bbox_inches='tight')
    print("Saved cache miss rate plot as 'cache_miss_rate.png'")

def main():
    print("\n=== Performance Analysis Script ===")
    print("Checking environment...")
    
    # Check if perf is installed
    try:
        subprocess.run(['perf', '--version'], capture_output=True, check=True)
    except subprocess.CalledProcessError:
        print("Error: 'perf' command not found. Please install the Linux perf tools.")
        return
    except FileNotFoundError:
        print("Error: 'perf' command not found. Please install the Linux perf tools.")
        return
        
    # First compile all programs
    if not compile_programs():
        print("Compilation failed. Exiting.")
        return

    # List your four programs here
    programs = ['mixed_workload', 'memory_bound', 'io_bound', 'cpu_bound']
    num_threads = 4
    
    print("\nRunning performance analysis...")
    # Run performance analysis
    results_df = run_perf_analysis(programs, num_threads)
    
    if results_df is None:
        print("Performance analysis failed. Exiting.")
        return
        
    print("Performance analysis completed.")
    
    # Create visualization plots
    create_performance_plots(results_df)
    
    # Print summary statistics
    print("\nPerformance Summary:")
    print(results_df.describe())

if __name__ == "__main__":
    main()