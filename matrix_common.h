#ifndef MATRIX_COMMON_H
#define MATRIX_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <omp.h>

#define MATRIX_SIZE 1000

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

#endif // MATRIX_COMMON_H
