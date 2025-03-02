#include "matrix_common.h"

// Global matrices
int (*A)[MATRIX_SIZE];
int (*B)[MATRIX_SIZE];
int (*C)[MATRIX_SIZE];

// Simple OpenMP matrix multiplication
void multiply_matrices_openmp_simple(int num_threads) {
    omp_set_num_threads(num_threads);
    
    double start_time = omp_get_wtime();
    
    #pragma omp parallel for
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            int sum = 0;
            for (int k = 0; k < MATRIX_SIZE; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
    
    double end_time = omp_get_wtime();
    printf("OpenMP simple multiplication completed in %.4f seconds\n", end_time - start_time);
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
    
    printf("Running simple OpenMP with %d threads\n", num_threads);
    multiply_matrices_openmp_simple(num_threads);

    printf("Writing result to matrix_C_openmp_simple.txt\n");
    write_matrix_to_file("matrix_C_openmp_simple.txt", C);

    // Calculate total program execution time
    clock_t program_end = clock();
    double total_time = (double)(program_end - program_start) / CLOCKS_PER_SEC;
    printf("Total program execution time: %.2f seconds\n", total_time);

    free(A);
    free(B);
    free(C);

    return 0;
}
