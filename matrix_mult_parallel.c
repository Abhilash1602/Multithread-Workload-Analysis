#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <pthread.h>

#define MATRIX_SIZE 1000
#define BUFFER_SIZE 10
#define MIN(a,b) ((a) < (b) ? (a) : (b))

typedef struct {
    int start_row;
    int end_row;
} Task;

typedef struct {
    Task buffer[BUFFER_SIZE];
    int in;
    int out;
    int count;
    int done;
    pthread_mutex_t mutex;
    pthread_cond_t not_full;
    pthread_cond_t not_empty;
} SharedBuffer;

typedef struct {
    int chunks_processed;
    double duration;
    double throughput;
} ThreadStats;

typedef struct {
    int thread_id;
    int num_threads;
    SharedBuffer* shared_buffer;
    ThreadStats* stats;  // Add stats pointer
} ThreadArgs;

int (*A)[MATRIX_SIZE];
int (*B)[MATRIX_SIZE];
int (*C)[MATRIX_SIZE];

SharedBuffer* init_shared_buffer() {
    SharedBuffer* buf = (SharedBuffer*)malloc(sizeof(SharedBuffer));
    buf->in = buf->out = buf->count = buf->done = 0;
    pthread_mutex_init(&buf->mutex, NULL);
    pthread_cond_init(&buf->not_full, NULL);
    pthread_cond_init(&buf->not_empty, NULL);
    return buf;
}

void* producer(void* arg) {
    ThreadArgs* args = (ThreadArgs*)arg;
    SharedBuffer* buf = args->shared_buffer;
    int rows_per_thread = MATRIX_SIZE / args->num_threads;
    int start = args->thread_id * rows_per_thread;
    int end = (args->thread_id == args->num_threads - 1) ? MATRIX_SIZE : start + rows_per_thread;

    for (int i = start; i < end; i += MATRIX_SIZE/100) {  // Split into smaller tasks
        Task task = {
            .start_row = i,
            .end_row = MIN(i + MATRIX_SIZE/100, end)
        };

        pthread_mutex_lock(&buf->mutex);
        while (buf->count == BUFFER_SIZE) {
            pthread_cond_wait(&buf->not_full, &buf->mutex);
        }
        
        buf->buffer[buf->in] = task;
        buf->in = (buf->in + 1) % BUFFER_SIZE;
        buf->count++;
        
        pthread_cond_signal(&buf->not_empty);
        pthread_mutex_unlock(&buf->mutex);
    }

    pthread_mutex_lock(&buf->mutex);
    buf->done = 1;
    pthread_cond_broadcast(&buf->not_empty);
    pthread_mutex_unlock(&buf->mutex);

    return NULL;
}

void* consumer(void* arg) {
    ThreadArgs* args = (ThreadArgs*)arg;
    SharedBuffer* buf = args->shared_buffer;
    int chunks_processed = 0;
    clock_t start = clock();

    while (1) {
        pthread_mutex_lock(&buf->mutex);
        while (buf->count == 0) {
            if (buf->done) {
                clock_t end = clock();
                double duration = (double)(end - start) / CLOCKS_PER_SEC;
                args->stats[args->thread_id].chunks_processed = chunks_processed;
                args->stats[args->thread_id].duration = duration;
                args->stats[args->thread_id].throughput = chunks_processed / duration;
                pthread_mutex_unlock(&buf->mutex);
                return NULL;
            }
            pthread_cond_wait(&buf->not_empty, &buf->mutex);
        }

        Task task = buf->buffer[buf->out];
        buf->out = (buf->out + 1) % BUFFER_SIZE;
        buf->count--;
        chunks_processed++;

        pthread_cond_signal(&buf->not_full);
        pthread_mutex_unlock(&buf->mutex);

        // Process the matrix multiplication task
        for (int i = task.start_row; i < task.end_row; i++) {
            for (int j = 0; j < MATRIX_SIZE; j++) {
                int sum = 0;
                for (int k = 0; k < MATRIX_SIZE; k++) {
                    sum += A[i][k] * B[k][j];
                }
                C[i][j] = sum;
            }
        }
    }
}

void multiply_matrices_parallel(int num_pairs) {
    pthread_t* producers = malloc(num_pairs * sizeof(pthread_t));
    pthread_t* consumers = malloc(num_pairs * sizeof(pthread_t));
    ThreadArgs* thread_args = malloc(num_pairs * sizeof(ThreadArgs));
    SharedBuffer** buffers = malloc(num_pairs * sizeof(SharedBuffer*));
    ThreadStats* thread_stats = malloc(num_pairs * sizeof(ThreadStats));

    // Create thread pairs
    for (int i = 0; i < num_pairs; i++) {
        buffers[i] = init_shared_buffer();
        thread_args[i].thread_id = i;
        thread_args[i].num_threads = num_pairs;
        thread_args[i].shared_buffer = buffers[i];
        thread_args[i].stats = thread_stats;

        pthread_create(&producers[i], NULL, producer, &thread_args[i]);
        pthread_create(&consumers[i], NULL, consumer, &thread_args[i]);
    }

    // Wait for all threads to complete
    for (int i = 0; i < num_pairs; i++) {
        pthread_join(producers[i], NULL);
        pthread_join(consumers[i], NULL);
        
        // Print thread statistics
        printf("\nThread Pair %d:\n", i);
        printf("Chunks processed: %d\n", thread_stats[i].chunks_processed);
        printf("Duration: %.2f seconds\n", thread_stats[i].duration);
        printf("Throughput: %.2f chunks/second\n", thread_stats[i].throughput);
        
        pthread_mutex_destroy(&buffers[i]->mutex);
        pthread_cond_destroy(&buffers[i]->not_full);
        pthread_cond_destroy(&buffers[i]->not_empty);
        free(buffers[i]);
    }

    free(producers);
    free(consumers);
    free(thread_args);
    free(buffers);
    free(thread_stats);
}

// Write matrix to file (space-separated format)
void write_matrix_to_file(const char* filename, int (*matrix)[MATRIX_SIZE]) {
    FILE* file = fopen(filename, "w");  // Changed to text mode
    if (!file) {
        perror("Error opening file for writing");
        return;
    }

    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            fprintf(file, "%d", matrix[i][j]);
            if (j < MATRIX_SIZE - 1) {
                fprintf(file, " ");  // Add space between numbers
            }
        }
        fprintf(file, "\n");  // New line at end of each row
    }
    fclose(file);
}

// Read matrix from text file (space-separated format)
void read_matrix_from_text_file(const char* filename, int (*matrix)[MATRIX_SIZE]) {
    FILE* file = fopen(filename, "r");  // Open in text mode
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
    int num_threads;
    printf("Enter the number of producer-consumer pairs: ");
    scanf("%d", &num_threads);
    
    if (num_threads < 1 || num_threads > 8) {
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

    printf("Multiplying matrices\n");
    
    // Start timing the entire program
    clock_t program_start = clock();
    
    // Matrix multiplication timing
    clock_t mult_start = clock();
    multiply_matrices_parallel(num_threads);
    clock_t mult_end = clock();
    
    double mult_time = (double)(mult_end - mult_start) / CLOCKS_PER_SEC;
    printf("Matrix multiplication completed in %f seconds\n", mult_time);
    
    printf("Writing result to matrix_C_int.txt\n");
    write_matrix_to_file("matrix_C_int.txt", C);

    // Calculate total program execution time
    clock_t program_end = clock();
    double total_time = (double)(program_end - program_start) / CLOCKS_PER_SEC;
    printf("\nTotal program execution time: %.2f seconds\n", total_time);

    free(A);
    free(B);
    free(C);

    return 0;
}
