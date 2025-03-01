#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>

#define MATRIX_SIZE 1000
#define BUFFER_SIZE 32
#define CHUNK_SIZE 16  // Number of rows to process in one task

// Work item structure
typedef struct {
    int start_row;
    int end_row;
} WorkItem;

// Circular buffer structure
typedef struct {
    WorkItem buffer[BUFFER_SIZE];
    int in;
    int out;
    int count;
    pthread_mutex_t mutex;
    pthread_cond_t not_full;
    pthread_cond_t not_empty;
    int done;
} CircularBuffer;

// Thread pair structure
typedef struct {
    int pair_id;
    CircularBuffer *buffer;
    int (*A)[MATRIX_SIZE];
    int (*B)[MATRIX_SIZE];
    int (*C)[MATRIX_SIZE];
    pthread_mutex_t *result_mutex;
    struct timespec start_time;
    struct timespec end_time;
    int items_processed;
    int total_pairs; // <-- new field
} ThreadPair;

// Initialize buffer
void init_buffer(CircularBuffer *buffer) {
    buffer->in = 0;
    buffer->out = 0;
    buffer->count = 0;
    buffer->done = 0;
    pthread_mutex_init(&buffer->mutex, NULL);
    pthread_cond_init(&buffer->not_full, NULL);
    pthread_cond_init(&buffer->not_empty, NULL);
}

// Producer thread function
void *producer(void *arg) {
    ThreadPair *pair = (ThreadPair *)arg;
    CircularBuffer *buffer = pair->buffer;

    int rows_per_pair = MATRIX_SIZE / pair->total_pairs;
    int start = pair->pair_id * rows_per_pair;
    int end = (pair->pair_id == pair->total_pairs - 1) ? MATRIX_SIZE 
                                                      : start + rows_per_pair;

    for (int start_row = start; start_row < end; start_row += CHUNK_SIZE) {
        pthread_mutex_lock(&buffer->mutex);
        
        while (buffer->count == BUFFER_SIZE) {
            pthread_cond_wait(&buffer->not_full, &buffer->mutex);
        }
        
        // Create work item
        WorkItem item;
        item.start_row = start_row;
        item.end_row = start_row + CHUNK_SIZE < end ? start_row + CHUNK_SIZE : end;
        
        buffer->buffer[buffer->in] = item;
        buffer->in = (buffer->in + 1) % BUFFER_SIZE;
        buffer->count++;
        
        pthread_cond_signal(&buffer->not_empty);
        pthread_mutex_unlock(&buffer->mutex);
    }
    
    // Signal completion
    pthread_mutex_lock(&buffer->mutex);
    buffer->done = 1;
    pthread_cond_broadcast(&buffer->not_empty);
    pthread_mutex_unlock(&buffer->mutex);
    
    return NULL;
}

// Consumer thread function
void *consumer(void *arg) {
    ThreadPair *pair = (ThreadPair *)arg;
    CircularBuffer *buffer = pair->buffer;
    
    while (1) {
        pthread_mutex_lock(&buffer->mutex);
        
        while (buffer->count == 0) {
            if (buffer->done) {
                pthread_mutex_unlock(&buffer->mutex);
                return NULL;
            }
            pthread_cond_wait(&buffer->not_empty, &buffer->mutex);
        }
        
        // Get work item
        WorkItem item = buffer->buffer[buffer->out];
        buffer->out = (buffer->out + 1) % BUFFER_SIZE;
        buffer->count--;
        
        pthread_cond_signal(&buffer->not_full);
        pthread_mutex_unlock(&buffer->mutex);
        
        // Process the chunk of matrix multiplication
        for (int i = item.start_row; i < item.end_row; i++) {
            for (int j = 0; j < MATRIX_SIZE; j++) {
                int sum = 0;
                for (int k = 0; k < MATRIX_SIZE; k++) {
                    sum += pair->A[i][k] * pair->B[k][j];
                }
                pthread_mutex_lock(&pair->result_mutex[i]);
                pair->C[i][j] = sum;
                pthread_mutex_unlock(&pair->result_mutex[i]);
            }
        }
        
        pair->items_processed++;
    }
}

void multiply_matrices(int (*A)[MATRIX_SIZE], int (*B)[MATRIX_SIZE], int (*C)[MATRIX_SIZE]) {
    int num_pairs;
    printf("Enter the number of producer-consumer pairs: ");
    scanf("%d", &num_pairs);
    
    ThreadPair *pairs = malloc(num_pairs * sizeof(ThreadPair));
    CircularBuffer *buffers = malloc(num_pairs * sizeof(CircularBuffer));
    pthread_t *producers = malloc(num_pairs * sizeof(pthread_t));
    pthread_t *consumers = malloc(num_pairs * sizeof(pthread_t));
    
    // Create mutex array for result matrix
    pthread_mutex_t *result_mutex = malloc(MATRIX_SIZE * sizeof(pthread_mutex_t));
    for (int i = 0; i < MATRIX_SIZE; i++) {
        pthread_mutex_init(&result_mutex[i], NULL);
    }
    
    // Initialize and start threads
    for (int i = 0; i < num_pairs; i++) {
        init_buffer(&buffers[i]);
        pairs[i].pair_id = i;
        pairs[i].buffer = &buffers[i];
        pairs[i].A = A;
        pairs[i].B = B;
        pairs[i].C = C;
        pairs[i].result_mutex = result_mutex;
        pairs[i].items_processed = 0;
        pairs[i].total_pairs = num_pairs; // <-- pass total pairs to each pair
        
        clock_gettime(CLOCK_MONOTONIC, &pairs[i].start_time);
        pthread_create(&producers[i], NULL, producer, &pairs[i]);
        pthread_create(&consumers[i], NULL, consumer, &pairs[i]);
    }
    
    // Wait for threads to complete
    for (int i = 0; i < num_pairs; i++) {
        pthread_join(producers[i], NULL);
        pthread_join(consumers[i], NULL);
        clock_gettime(CLOCK_MONOTONIC, &pairs[i].end_time);
        
        // Print performance metrics
        double duration = (pairs[i].end_time.tv_sec - pairs[i].start_time.tv_sec) +
                         (pairs[i].end_time.tv_nsec - pairs[i].start_time.tv_nsec) / 1e9;
        printf("Thread Pair %d:\n", i);
        printf("Chunks processed: %d\n", pairs[i].items_processed);
        printf("Duration: %.2f seconds\n", duration);
        printf("Throughput: %.2f chunks/second\n\n", pairs[i].items_processed / duration);
        
        // Cleanup
        pthread_mutex_destroy(&buffers[i].mutex);
        pthread_cond_destroy(&buffers[i].not_full);
        pthread_cond_destroy(&buffers[i].not_empty);
    }
    
    // Cleanup result mutexes
    for (int i = 0; i < MATRIX_SIZE; i++) {
        pthread_mutex_destroy(&result_mutex[i]);
    }
    
    free(result_mutex);
    free(pairs);
    free(buffers);
    free(producers);
    free(consumers);
}

// Write matrix to file (space-separated format)
void write_matrix_to_file(const char* filename, int (*matrix)[MATRIX_SIZE]) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        perror("Error opening file for writing");
        return;
    }

    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            fprintf(file, "%d", matrix[i][j]);
            if (j < MATRIX_SIZE - 1) {
                fprintf(file, " ");
            }
        }
        fprintf(file, "\n");
    }
    fclose(file);
}

// Read matrix from text file (space-separated format)
void read_matrix_from_text_file(const char* filename, int (*matrix)[MATRIX_SIZE]) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file for reading");
        return;
    }

    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            if (fscanf(file, "%d", &matrix[i][j]) != 1) {
                printf("Error reading value at position (%d,%d)\n", i, j);
                fclose(file);
                return;
            }
        }
    }
    fclose(file);
}

int main() {
    struct timespec program_start, program_end;
    clock_gettime(CLOCK_MONOTONIC, &program_start);
    
    // Allocate matrices properly
    int (*A)[MATRIX_SIZE] = calloc(MATRIX_SIZE, sizeof(*A));
    int (*B)[MATRIX_SIZE] = calloc(MATRIX_SIZE, sizeof(*B));
    int (*C)[MATRIX_SIZE] = calloc(MATRIX_SIZE, sizeof(*C));

    if (!A || !B || !C) {
        perror("Memory allocation failed");
        // Clean up any successful allocations
        if (A) free(A);
        if (B) free(B);
        if (C) free(C);
        return 1;
    }

    // Initialize result matrix to zero
    memset(C, 0, MATRIX_SIZE * sizeof(*C));

    // Read input matrices with error checking
    printf("Reading matrix A\n");
    FILE* file_A = fopen("matrix_A_int.txt", "r");
    if (!file_A) {
        perror("Error opening matrix_A_int.txt");
        goto cleanup;
    }
    read_matrix_from_text_file("matrix_A_int.txt", A);
    fclose(file_A);

    printf("Reading matrix B\n");
    FILE* file_B = fopen("matrix_B_int.txt", "r");
    if (!file_B) {
        perror("Error opening matrix_B_int.txt");
        goto cleanup;
    }
    read_matrix_from_text_file("matrix_B_int.txt", B);
    fclose(file_B);

    // Multiply matrices with timing
    printf("Multiplying matrices\n");
    clock_t start_time = clock();
    multiply_matrices(A, B, C);
    clock_t end_time = clock();
    
    double execution_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Matrix multiplication completed in %f seconds\n", execution_time);

    // Write result to file
    printf("Writing result to matrix_C_int.txt\n");
    write_matrix_to_file("matrix_C_int.txt", C);

    clock_gettime(CLOCK_MONOTONIC, &program_end);
    double total_time = (program_end.tv_sec - program_start.tv_sec) +
                       (program_end.tv_nsec - program_start.tv_nsec) / 1e9;
    printf("\nTotal program execution time: %.2f seconds\n", total_time);

cleanup:
    // Clean up
    free(A);
    free(B);
    free(C);

    return 0;
}
