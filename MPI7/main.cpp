#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

#define NRA 1500 /* number of rows in matrix A */
#define NCA 1500 /* number of columns in matrix A */
#define NCB 1500 /* number of columns in matrix B */
#define MASTER 0 /* taskid of first task */
#define FROM_MASTER 1 /* setting a message type */
#define FROM_WORKER 10 /* setting a message type */

double** create_dynamic_arr(int rows, int cols) {
    double* mem = (double*)malloc(rows * cols * sizeof(double));
    double** A = (double**)malloc(rows * sizeof(double*));
    A[0] = mem;
    for (int i = 1; i < rows; i++) A[i] = A[i - 1] + cols;
    return A;
}

void free_dynamic_arr(double** arr) {
    free(arr[0]);
    free(arr);
}

int main(int argc, char* argv[]) {
    int numtasks,
            taskid,
            numworkers,
            i, j, rc = 0;
    double time;

    double** a = create_dynamic_arr(NRA, NCA);
    double** b = create_dynamic_arr(NCA, NCB);
    double** c = create_dynamic_arr(NRA, NCB);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);

    double** a_rows = create_dynamic_arr(NRA / numtasks, NRA);
    double** c_rows = create_dynamic_arr(NRA / numtasks, NRA);

    if (numtasks < 2) {
        printf("Need at least two MPI tasks. Quitting...\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(1);
    }
    numworkers = numtasks - 1;

    if (taskid == MASTER) {
        printf("mpi_mm has started with %d tasks.\n", numtasks);

        for (i = 0; i < NRA; i++)
            for (j = 0; j < NCA; j++)
                a[i][j] = 10;
        for (i = 0; i < NCA; i++)
            for (j = 0; j < NCB; j++)
                b[i][j] = 10;

        time = MPI_Wtime();
    }

    MPI_Scatter(
            *a,
            NRA * NRA / numtasks,
            MPI_DOUBLE,
            *a_rows,
            NRA * NRA / numtasks,
            MPI_DOUBLE,
            MASTER,
            MPI_COMM_WORLD);

    MPI_Bcast(
            *b,
            NRA * NRA,
            MPI_DOUBLE,
            MASTER,
            MPI_COMM_WORLD);

    for (int i = 0; i < NRA / numtasks; i++) {
        for (int j = 0; j < NRA; j++) {
            c_rows[i][j] = 0.0e0;
        }
    }

    for (int i = 0; i < NRA / numtasks; i++) {
        for (int k = 0; k < NRA; k++) {
            for (int j = 0; j < NRA; j++) {
                c_rows[i][j] += a_rows[i][k] * b[k][j];
            }
        }
    }

    MPI_Gather(
            *c_rows,
            (NRA / numtasks) * NRA,
            MPI_DOUBLE,
            *c,
            (NRA / numtasks) * NRA,
            MPI_DOUBLE,
            MASTER,
            MPI_COMM_WORLD);

    if (taskid == MASTER) {
        time = MPI_Wtime() - time;

        printf("\nTotal time: %.2f\n", time);
    }

    free_dynamic_arr(a);
    free_dynamic_arr(b);
    free_dynamic_arr(c);
    free_dynamic_arr(a_rows);
    free_dynamic_arr(c_rows);

    MPI_Finalize();
}
