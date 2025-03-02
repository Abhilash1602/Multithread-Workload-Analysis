#include "matrix_common.h"

// Global matrices
int (*A)[MATRIX_SIZE];
int (*B)[MATRIX_SIZE];
int (*C)[MATRIX_SIZE];

// Block-based OpenMP matrix multiplication with tasks
void multiply_matrices_openmp_tasks(int num_threads, int block_size) {
    omp_set_num_threads(num_threads);
    
    double start_time = omp_get_wtime();
    
    // Initialize result matrix to zeros
    memset(C, 0, MATRIX_SIZE * MATRIX_SIZE * sizeof(int));
    
    // Calculate number of blocks
    int num_blocks = (MATRIX_SIZE + block_size - 1) / block_size;
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            // Create tasks for each block
            for (int bi = 0; bi < num_blocks; bi++) {
                for (int bj = 0; bj < num_blocks; bj++) {
                    #pragma omp task firstprivate(bi, bj)
                    {
                        // Calculate block boundaries
                        int i_start = bi * block_size;
                        int i_end = (bi + 1) * block_size < MATRIX_SIZE ? (bi + 1) * block_size : MATRIX_SIZE;
                        int j_start = bj * block_size;
                        int j_end = (bj + 1) * block_size < MATRIX_SIZE ? (bj + 1) * block_size : MATRIX_SIZE;
                        
                        // Multiply for this block
                        for (int i = i_start; i < i_end; i++) {
                            for (int j = j_start; j < j_end; j++) {
                                int sum = 0;
                                for (int k = 0; k < MATRIX_SIZE; k++) {
                                    sum += A[i][k] * B[k][j];
                                }
                                C[i][j] = sum;
                            }
                        }
                    }
                }
            }
        } // implicit barrier at the end of single region
    } // implicit barrier at the end of parallel region
    
    double end_time = omp_get_wtime();
    printf("OpenMP task-based multiplication completed in %.4f seconds with block size %d\n", 
           end_time - start_time, block_size);
}

int main() {
    int num_threads;
    printf("Enter the number of threads: ");
    scanf("%d", &num_threads);
    
    if (num_threads < 1) {
        printf("Invalid number of threads. Using 4 threads.\n");
        num_threads = 4;
    }

    // Allocate matrices
    A = malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(int));
    B = malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(int));
    C = malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(int));

    if (!A || !B || !C) {
        perror("Memory allocation failed");
        return 1;
    }

    memset(C, 0, MATRIX_SIZE * MATRIX_SIZE * sizeof(int));

    printf("Reading matrix A\n");
    read_matrix_from_text_file("matrix_A_int.txt", A);
    printf("Reading matrix B\n");
    read_matrix_from_text_file("matrix_B_int.txt", B);

    // Start timing the entire program
    clock_t program_start = clock();
    
    printf("Running task-based OpenMP with %d threads\n", num_threads);
    
    // Try different block sizes for best performance
    int block_sizes[] = {16, 32, 64, 128};
    for (int i = 0; i < sizeof(block_sizes)/sizeof(int); i++) {
        multiply_matrices_openmp_tasks(num_threads, block_sizes[i]);
    }

    // Use the result from the last execution (block_size=128)
    printf("Writing result to matrix_C_openmp_tasks.txt\n");
    write_matrix_to_file("matrix_C_openmp_tasks.txt", C);

    // Calculate total program execution time
    clock_t program_end = clock();
    double total_time = (double)(program_end - program_start) / CLOCKS_PER_SEC;
    printf("Total program execution time: %.2f seconds\n", total_time);

    free(A);
    free(B);
    free(C);

    return 0;
}
