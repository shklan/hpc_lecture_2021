#include <mpi.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
#include <immintrin.h>

using namespace std;

void matmul(vector<float> &A, vector<float> &B, vector<float> &C, int N, int size) {
// A: N/size * N
// B: N * N/size
// C: N/size * N/size
   const int m = N/size, k = N, n = N/size;

   const int kc = k/4;
   const int nc = n/4;
   const int mc = m/4;
   const int nr = nc/2;
   const int mr = mc/2;
// simple cache blocking
//#pragma omp parallel for
//  for (int i=0; i<N/size; i++)
//    for (int k=0; k<N; k++)
//      for (int j=0; j<N/size; j++) {
//        // __m256 Avec = __mm256_load_ps();
//        // __m256 Bvec = __mm256_load_ps();
//        // __m256 Cvec = __mm256_load_ps();
//        C[N/size*i+j] += A[N*i+k] * B[N/size*k+j];
//      }

#pragma omp parallel for
  for (int jc=0; jc<n; jc+=nc) {
    for (int pc=0; pc<k; pc+=kc) {
      float Bc[kc*nc];
      // pack into Bc
      for (int p=0; p<kc; p++) {
        for (int j=0; j<nc; j++) {
          Bc[p*nc+j] = B[(p+pc)*n + (j+jc)];
        }  
      }
      for (int ic=0; ic<m; ic+=mc) {
        float Ac[mc*kc], Cc[mc*nc];
        for (int i=0; i<mc; i++) {
          // pack into Ac
          for (int p=0; p<kc; p++) {
            Ac[i*kc+p] = A[(i+ic)*k + (p+pc)];
          }
          // Initialize Cc
          for (int j=0; j<nc; j++) {
            Cc[i*nc+j] = 0;
          }
        }
        for (int jr=0; jr<nc; jr+=nr) {
          for (int ir=0; ir<mc; ir+=mr) {
            // matmul
            for (int kr=0; kr<kc; kr++) {
              for (int i=ir; i<ir+mr; i++) {
                for (int j=jr; j<jr+nr; j++) {
                  Cc[i*nc+j] += Ac[i*kc+kr] * Bc[kr*nc+j];
                }
              }
            }
          }
        }
        // unpack from Cc
        for (int i=0; i<mc; i++) {
          for (int j=0; j<nc; j++) {
            C[(i+ic)*n + (j+jc)] += Cc[i*nc+j];
          }
        }
      }
    }
  }  
}

void printMatrix(vector<float> &M, int N) {
   printf("printMatrix:\n");
   for(int i=0; i<N; i++) {
     printf("[");
     for (int j=0; j<N; j++) {
       printf(" %lf", M[N*i+j]);
     }
     printf("]\n");
   }
}

int main(int argc, char** argv) {
  int size, rank;
  
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
//  const int N = 2;
  const int N = 1024;
  vector<float> A(N*N);
  vector<float> B(N*N);
  vector<float> C(N*N, 0);
  vector<float> subA(N*N/size, 0);
  vector<float> subB(N*N/size, 0);
  vector<float> subC(N*N/size, 0);
  vector<float> subsubC(N/size*N/size, 0);
  matmul(subA, subB, subsubC, N, size);

  vector<float> recv(N*N/size);

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
//    if (!rank) {
//      printf("subA [");
//      for(int i=0; i<N*N/size; i++) {
//        printf(" %lf", subA[i]);
//      }
//      printf("]\n");
//    }
    matmul(subA, subB, subsubC, N, size);
// assign perhaps correct
    for (int i=0; i<N/size; i++)
      for (int j=0; j<N/size; j++) {
        subC[N*i+j+offset] = subsubC[N/size*i+j];
        subsubC[N/size*i+j] = 0;
//        printf("rank: %d, %d <- %d\n", rank, N*i+j+offset, N/size*i+j);
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
//    if (!rank) {
//      printf("subB [");
//      for (int i=0; i<N*N/size; i++) {
//        printf(" %lf", subB[i]);
//      }
//      printf("]\n");
//      printf("subsubC [");
//      for (int i=0; i<N/size*N/size; i++) {
//        printf(" %lf", subsubC[i]);
//      }
//      printf("]\n");
//    } 
  }
  MPI_Allgather(&subC[0], N*N/size, MPI_FLOAT, &C[0], N*N/size, MPI_FLOAT, MPI_COMM_WORLD);
 
//  if (!rank) { 
//    printf("A\n");
//    printMatrix(A, N);
//    printf("B\n");
//    printMatrix(B, N);
//    printf("C\n");
//    printMatrix(C, N);
//  }

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
  MPI_Finalize();
}
