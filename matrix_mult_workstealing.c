#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <pthread.h>
#include <stdbool.h>

#define MATRIX_SIZE 1000
#define BLOCK_SIZE 32  // Block size for matrix multiplication
#define MAX_THREADS 32
#define TASK_QUEUE_SIZE 1024

// Task structure for work-stealing
typedef struct {
    int i_start;  // Starting row
    int i_end;    // Ending row
    int j_start;  // Starting column
    int j_end;    // Ending column
} Task;

// Work-stealing task queue
typedef struct {
    Task tasks[TASK_QUEUE_SIZE];
    int top;  // Points to the next empty slot
    int bottom; // Points to the first task
    pthread_mutex_t mutex;
} TaskQueue;

// Thread data structure
typedef struct {
    int thread_id;
    int num_threads;
    TaskQueue* local_queue;    // Thread's own task queue
    TaskQueue** all_queues;    // All thread queues for stealing
    double work_time;          // Time spent working
    int tasks_processed;       // Number of tasks processed
    int tasks_stolen;          // Number of tasks stolen
} ThreadData;

// Global matrices
int (*A)[MATRIX_SIZE];
int (*B)[MATRIX_SIZE];
int (*C)[MATRIX_SIZE];

// Initialize task queue
void init_task_queue(TaskQueue* queue) {
    queue->top = 0;
    queue->bottom = 0;
    pthread_mutex_init(&queue->mutex, NULL);
}

// Push task to local queue (thread-safe, but designed for single producer)
bool push_task(TaskQueue* queue, Task task) {
    pthread_mutex_lock(&queue->mutex);
    if ((queue->top + 1) % TASK_QUEUE_SIZE == queue->bottom) {
        // Queue is full
        pthread_mutex_unlock(&queue->mutex);
        return false;
    }
    
    queue->tasks[queue->top] = task;
    queue->top = (queue->top + 1) % TASK_QUEUE_SIZE;
    pthread_mutex_unlock(&queue->mutex);
    return true;
}

// Pop task from local queue
bool pop_task(TaskQueue* queue, Task* task) {
    pthread_mutex_lock(&queue->mutex);
    if (queue->top == queue->bottom) {
        // Queue is empty
        pthread_mutex_unlock(&queue->mutex);
        return false;
    }
    
    queue->top = (queue->top + TASK_QUEUE_SIZE - 1) % TASK_QUEUE_SIZE;
    *task = queue->tasks[queue->top];
    pthread_mutex_unlock(&queue->mutex);
    return true;
}

// Steal task from victim queue (from bottom of queue)
bool steal_task(TaskQueue* queue, Task* task) {
    pthread_mutex_lock(&queue->mutex);
    if (queue->top == queue->bottom) {
        // Queue is empty
        pthread_mutex_unlock(&queue->mutex);
        return false;
    }
    
    *task = queue->tasks[queue->bottom];
    queue->bottom = (queue->bottom + 1) % TASK_QUEUE_SIZE;
    pthread_mutex_unlock(&queue->mutex);
    return true;
}

// Process a single matrix multiplication block task
void process_task(Task* task) {
    for (int i = task->i_start; i < task->i_end; i++) {
        for (int j = task->j_start; j < task->j_end; j++) {
            int sum = 0;
            for (int k = 0; k < MATRIX_SIZE; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

// Worker thread function with work stealing
void* worker_thread(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    TaskQueue* my_queue = data->local_queue;
    Task task;
    bool idle = false;
    int idle_count = 0;
    clock_t start_time = clock();
    
    while (1) {
        // First, try to pop from own queue
        if (pop_task(my_queue, &task)) {
            process_task(&task);
            data->tasks_processed++;
            idle = false;
            idle_count = 0;
            continue;
        }
        
        // If local queue is empty, try to steal
        bool stole = false;
        for (int i = 0; i < data->num_threads; i++) {
            if (i == data->thread_id) continue;  // Don't steal from self
            
            if (steal_task(data->all_queues[i], &task)) {
                process_task(&task);
                data->tasks_processed++;
                data->tasks_stolen++;
                stole = true;
                idle = false;
                idle_count = 0;
                break;
            }
        }
        
        if (!stole) {
            // No work was found
            idle = true;
            idle_count++;
            
            // After trying to steal N times, check if all queues are empty
            if (idle_count > data->num_threads * 2) {
                bool all_empty = true;
                for (int i = 0; i < data->num_threads; i++) {
                    TaskQueue* q = data->all_queues[i];
                    pthread_mutex_lock(&q->mutex);
                    if (q->top != q->bottom) {
                        all_empty = false;
                    }
                    pthread_mutex_unlock(&q->mutex);
                    if (!all_empty) break;
                }
                
                if (all_empty) {
                    break; // Exit the loop if all queues are empty
                }
                
                // Reset idle count and try again
                idle_count = 0;
            }
            
            // Small delay to reduce CPU usage during stealing attempts
            struct timespec ts = {0, 100000}; // 100 microseconds
            nanosleep(&ts, NULL);
        }
    }
    
    clock_t end_time = clock();
    data->work_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    
    return NULL;
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

void multiply_matrices_workstealing(int num_threads) {
    // Calculate number of blocks
    int num_blocks_i = (MATRIX_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int num_blocks_j = (MATRIX_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int total_blocks = num_blocks_i * num_blocks_j;
    
    printf("Matrix divided into %d blocks (%dx%d)\n", total_blocks, num_blocks_i, num_blocks_j);
    
    // Initialize thread data and queues
    pthread_t threads[MAX_THREADS];
    ThreadData thread_data[MAX_THREADS];
    TaskQueue queues[MAX_THREADS];
    TaskQueue* queue_ptrs[MAX_THREADS];
    
    for (int i = 0; i < num_threads; i++) {
        init_task_queue(&queues[i]);
        queue_ptrs[i] = &queues[i];
        thread_data[i].thread_id = i;
        thread_data[i].num_threads = num_threads;
        thread_data[i].local_queue = &queues[i];
        thread_data[i].all_queues = queue_ptrs;
        thread_data[i].tasks_processed = 0;
        thread_data[i].tasks_stolen = 0;
    }
    
    // Create tasks and distribute them among threads (initially with cyclic distribution)
    int task_count = 0;
    for (int bi = 0; bi < num_blocks_i; bi++) {
        for (int bj = 0; bj < num_blocks_j; bj++) {
            Task task;
            task.i_start = bi * BLOCK_SIZE;
            task.i_end = (bi + 1) * BLOCK_SIZE < MATRIX_SIZE ? (bi + 1) * BLOCK_SIZE : MATRIX_SIZE;
            task.j_start = bj * BLOCK_SIZE;
            task.j_end = (bj + 1) * BLOCK_SIZE < MATRIX_SIZE ? (bj + 1) * BLOCK_SIZE : MATRIX_SIZE;
            
            // Initial distribution of tasks (round-robin)
            int target_thread = task_count % num_threads;
            push_task(&queues[target_thread], task);
            task_count++;
        }
    }
    
    printf("Created and distributed %d tasks among %d threads\n", task_count, num_threads);
    
    // Start worker threads
    clock_t mult_start = clock();
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, worker_thread, &thread_data[i]);
    }
    
    // Wait for threads to complete
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    clock_t mult_end = clock();
    
    double mult_time = (double)(mult_end - mult_start) / CLOCKS_PER_SEC;
    printf("Work-stealing matrix multiplication completed in %.4f seconds\n", mult_time);
    
    // Print statistics per thread
    printf("\nThread Statistics:\n");
    printf("Thread | Tasks Processed | Tasks Stolen | Work Time (s)\n");
    printf("------------------------------------------------------\n");
    
    int total_processed = 0;
    int total_stolen = 0;
    for (int i = 0; i < num_threads; i++) {
        printf("%6d | %14d | %12d | %12.4f\n", 
               i, 
               thread_data[i].tasks_processed, 
               thread_data[i].tasks_stolen,
               thread_data[i].work_time);
        total_processed += thread_data[i].tasks_processed;
        total_stolen += thread_data[i].tasks_stolen;
    }
    
    printf("------------------------------------------------------\n");
    printf("Total  | %14d | %12d | \n", total_processed, total_stolen);
    printf("------------------------------------------------------\n");
    
    // Clean up
    for (int i = 0; i < num_threads; i++) {
        pthread_mutex_destroy(&queues[i].mutex);
    }
}

int main() {
    int num_threads;
    printf("Enter the number of threads: ");
    scanf("%d", &num_threads);
    
    if (num_threads < 1 || num_threads > MAX_THREADS) {
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
    
    printf("Running with %d threads\n", num_threads);
    
    // Matrix multiplication timing
    multiply_matrices_workstealing(num_threads);

    printf("Writing result to matrix_C_workstealing.txt\n");
    write_matrix_to_file("matrix_C_workstealing.txt", C);

    // Calculate total program execution time
    clock_t program_end = clock();
    double total_time = (double)(program_end - program_start) / CLOCKS_PER_SEC;
    printf("\nTotal program execution time: %.2f seconds\n", total_time);

    free(A);
    free(B);
    free(C);

    return 0;
}
