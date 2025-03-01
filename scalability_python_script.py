import subprocess
import time

def run_perf(output_file, threads_list, executable):
    with open(output_file, 'a') as f:
        for threads in threads_list:
            f.write(f"Running with {threads} threads...\n")
            
            # Run perf command with input redirection for thread count
            process = subprocess.Popen(
                ["sudo", "perf", "stat", "-e",
                 "cycles,instructions,cache-references,cache-misses,branch-misses,"
                 "task-clock,context-switches,cpu-migrations,page-faults,L1-dcache-loads,"
                 "L1-dcache-load-misses,LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses,"
                 "mem-loads,mem-stores,sched:sched_stat_sleep",
                 "--", executable],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            
            # Provide input for the number of threads
            stdout, _ = process.communicate(input=str(threads) + "\n")
            
            # Append output to file
            f.write(stdout + "\n")
            f.write("---------------------------------------------------\n")
            
            time.sleep(1)  # Avoid rapid successive runs

if __name__ == "__main__":
    output_file = "output_c_p.txt"
    threads_list = [1, 2, 3, 4, 5, 6 , 7, 8]  # Update this if needed
    executable = "./matrix_mult_p"  # Update this if needed
    
    run_perf(output_file, threads_list, executable)
    print(f"Performance analysis completed. Results saved in {output_file}")
