#include <mpi.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
using namespace std;

#define M 1024

__global__ void matmul(float *A, float *B, float *C, int N, int size) {
  int Nc = N/size;
//  int i = blockIdx.x
//  int j = threadIdx.x
  int raw_index = blockIdx.x * blockDim.x + threadIdx.x;
  int i = raw_index / Nc;
  int j = raw_index % Nc;
  float sum = 0;

  for (int k=0; k<N; k++) {
    sum += A[N*i+k] * B[N/size*k+j];
  }
  C[Nc*i+j] = sum;
}

void cudaCheckError() {
  cudaError_t err=cudaGetLastError();
  if (err!=cudaSuccess) {
    fprintf(stderr, "CUDA Failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(err));
    exit(-1);
  }
}

int main(int argc, char** argv) {
  int size, rank;
  
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  const int N = 1024;
  const int Nc = N/size;
  vector<float> A(N*N);
  vector<float> B(N*N);
  vector<float> C(N*N, 0);
//  vector<float> subA(N*N/size);
//  vector<float> subB(N*N/size);
  vector<float> subC(N*N/size, 0);
  vector<float> recv(N*N/size);
  float *subA, *subB, *subsubC;
  cudaMallocManaged(&subA, N*N/size * sizeof(float));
  cudaMallocManaged(&subB, N*N/size * sizeof(float));
  cudaMallocManaged(&subsubC, N/size*N/size * sizeof(float));

  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      A[N*i+j] = drand48();
      B[N*i+j] = drand48();
    }
  }
  
  int offset = N/size*rank;
  for (int i=0; i<N/size; i++)
    for (int j=0; j<N; j++)
      subA[N*i+j] = A[N*(i+offset)+j];
  for (int i=0; i<N; i++)
    for (int j=0; j<N/size; j++)
      subB[N/size*i+j] = B[N*i+j+offset];
  
  int recv_from = (rank + 1) % size;
  int send_to = (rank - 1 + size) % size;

  double comp_time = 0, comm_time = 0;
  for(int irank=0; irank<size; irank++) {
    auto tic = chrono::steady_clock::now();
    offset = N/size*((rank+irank) % size);
    matmul<<<Nc*Nc/M, M>>>(subA, subB, subsubC, N, size);
    cudaCheckError();
    cudaDeviceSynchronize();
#pragma omp parallel for collapse(2)
    for (int i=0; i<N/size; i++)
      for (int j=0; j<N/size; j++) {
        subC[N*i+j+offset] = subsubC[Nc*i+j];
        subsubC[Nc*i+j] = 0;
      }
    auto toc = chrono::steady_clock::now();
    comp_time += chrono::duration<double>(toc - tic).count();
    MPI_Request request[2];
    MPI_Isend(&subB[0], N*N/size, MPI_FLOAT, send_to, 0, MPI_COMM_WORLD, &request[0]);
    MPI_Irecv(&recv[0], N*N/size, MPI_FLOAT, recv_from, 0, MPI_COMM_WORLD, &request[1]);
    MPI_Waitall(2, request, MPI_STATUS_IGNORE);
    for (int i=0; i<N*N/size; i++)
      subB[i] = recv[i];
    tic = chrono::steady_clock::now();
    comm_time += chrono::duration<double>(tic - toc).count();
  }
  MPI_Allgather(&subC[0], N*N/size, MPI_FLOAT, &C[0], N*N/size, MPI_FLOAT, MPI_COMM_WORLD);
  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      for (int k=0; k<N; k++)
        C[N*i+j] -= A[N*i+k] * B[N*k+j];
  double err = 0;
  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      err += fabs(C[N*i+j]);
  if(rank==0) {
    double time = comp_time+comm_time;
    printf("N    : %d\n",N);
    printf("comp : %lf s\n", comp_time);
    printf("comm : %lf s\n", comm_time);
    printf("total: %lf s (%lf GFlops)\n",time,2.*N*N*N/time/1e9);
    printf("error: %lf\n",err/N/N);
  }
  cudaFree(subA);
  cudaFree(subB);
  cudaFree(subsubC);
  MPI_Finalize();
}
