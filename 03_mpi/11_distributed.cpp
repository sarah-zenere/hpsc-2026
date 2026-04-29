#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <mpi.h>
//manually editing this bc vscode won't commit :(

struct Body {
  double x, y, m, fx, fy;
};

int main(int argc, char** argv) {
  const int N = 20;

  MPI_Init(&argc, &argv);

  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  //change N
  if (N % size != 0) {
    if (rank == 0) {
      printf("Error: N must be divisible by number of processes\n");
    }
    MPI_Finalize();
    return 1;
  }

  const int local_n = N / size;

  Body ibody[local_n];     
  Body jbody[local_n];     
  Body send_buf[local_n];  //ring around the rosie

  srand48(rank);

  //init bodies
  for (int i = 0; i < local_n; i++) {
    ibody[i].x = drand48();
    ibody[i].y = drand48();
    ibody[i].m = drand48();

    ibody[i].fx = 0.0;
    ibody[i].fy = 0.0;

    //ring 
    jbody[i] = ibody[i];
    send_buf[i] = ibody[i];
  }

  //MPI dt for Body
  MPI_Datatype MPI_BODY;
  MPI_Type_contiguous(5, MPI_DOUBLE, &MPI_BODY);
  MPI_Type_commit(&MPI_BODY);

  //window exposing jbody
  MPI_Win win;
  MPI_Win_create(
      jbody,
      local_n * sizeof(Body),
      sizeof(Body),
      MPI_INFO_NULL,
      MPI_COMM_WORLD,
      &win
  );

  int next = (rank + 1) % size;

  for (int irank = 0; irank < size; irank++) {

    //jbody contents
    for (int i = 0; i < local_n; i++) {
      for (int j = 0; j < local_n; j++) {

        double rx = ibody[i].x - jbody[j].x;
        double ry = ibody[i].y - jbody[j].y;

        double r = std::sqrt(rx * rx + ry * ry);

        if (r > 1e-15) {
          ibody[i].fx -= rx * jbody[j].m / (r * r * r);
          ibody[i].fy -= ry * jbody[j].m / (r * r * r);
        }
      }
    }

   
    if (irank < size - 1) {

      //epoch
      MPI_Win_fence(0, win);

     
      MPI_Put(
          send_buf,     
          local_n,         
          MPI_BODY,       
          next,           
          0,              
          local_n,         
          MPI_BODY,       
          win
      );

      //
      MPI_Win_fence(0, win);

      for (int i = 0; i < local_n; i++) {
        send_buf[i] = jbody[i];
      }
    }
  }

  //results in rank order
  for (int irank = 0; irank < size; irank++) {

    MPI_Barrier(MPI_COMM_WORLD);

    if (irank == rank) {
      for (int i = 0; i < local_n; i++) {
        printf("%d %g %g\n",
               i + rank * local_n,
               ibody[i].fx,
               ibody[i].fy);
      }
    }
  }

  MPI_Win_free(&win);
  MPI_Type_free(&MPI_BODY);

  MPI_Finalize();

  return 0;
} //ssss
