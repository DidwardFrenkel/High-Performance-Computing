#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <sys/time.h>

int main(int argc,char* argv[])
{
  int rank,nprocs,N;	// N determines how often message is sent around the
			// ring.

  struct timeval t1,t2;

  if (argc < 2)
  {
    N = 1;		//set N = 1 as default if no N is specified
    /*printf("Need number of iterations for this task.");
    abort();*/
  } else {
    N = atoi(argv[1]);
  }
  gettimeofday(&t1,NULL);
  MPI_Status status;

  int message_in=0, message_out=0;

  int i;
  MPI_Init(&argc,&argv);

  for (i = 0;i<N;i++) {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  int destination, origin,tag = 99;

  if (rank == 0) {
    destination = nprocs==1 ? 0:1;
    origin = nprocs-1;
    
    MPI_Send(&message_out,1, MPI_INT, destination, tag, MPI_COMM_WORLD);
    MPI_Recv(&message_in, 1, MPI_INT, origin, tag, MPI_COMM_WORLD, &status);
    message_out = message_in;
  } else if (rank == nprocs - 1) {
    origin = rank - 1;
    MPI_Recv(&message_in, 1, MPI_INT, origin, tag, MPI_COMM_WORLD, &status);
    message_out = message_in + rank;
    MPI_Send(&message_out,1, MPI_INT, 0, tag, MPI_COMM_WORLD);
  } else {
    destination = rank+1;
    origin = rank - 1;

    MPI_Recv(&message_in, 1, MPI_INT, origin, tag, MPI_COMM_WORLD, &status);
    message_out = message_in + rank;
    MPI_Send(&message_out,1, MPI_INT, destination, tag, MPI_COMM_WORLD);
  }

  printf("rank %d received from %d the message %d\n",rank,origin,message_in); 
  }
  MPI_Finalize();

  if (i == N && rank == nprocs - 1) {
  gettimeofday(&t2,NULL);
  printf("Latency: %lf\n",((double)(t2.tv_usec - t1.tv_usec)/1000000 + (double)(t2.tv_sec - t1.tv_sec))/(nprocs*N));
  }
  return 0;
}
