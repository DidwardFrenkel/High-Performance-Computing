/* Parallel sample sort
 */
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <stdlib.h>


static int compare(const void *a, const void *b)
{
  int *da = (int *)a;
  int *db = (int *)b;

  if (*da > *db)
    return 1;
  else if (*da < *db)
    return -1;
  else
    return 0;
}

int main( int argc, char *argv[])
{
  int rank;
  int i, tag = 99,N,P;
  int *vec;
  MPI_Status status;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &P);

  /* Number of random numbers per processor (this should be increased
   * for actual tests or could be made a passed in through the command line */
  N = 100;

  vec = calloc(N, sizeof(int));
  /* seed random number generator differently on every core */
  srand48((unsigned int) rank);

  /* fill vector with random integers */
  for (i = 0; i < N; ++i) {
    vec[i] = rand();
  }

  /* sort locally */
  qsort(vec, N, sizeof(int), compare);

  //With only one processor, this is just quicksort in serial.
  if (P == 1) {
    MPI_Finalize();
    for (i = 0;i < N; i++) {
      printf("%d\n",vec[i]);
    }
    return 0;
  }

  /* randomly sample s entries from vector or select local splitters,
   * i.e., every N/P-th entry of the sorted vector */
  int* s_entries = calloc(P-1,sizeof(int));
  int divisor = N/P;
  if (N % P != 0) divisor = N/P + 1;
  for (i = 1; i<P;i++) {
    s_entries[i-1] = vec[divisor*i-1];
  }

  /* every processor communicates the selected entries
   * to the root processor rank = 0.*/
  MPI_Send(s_entries,P-1,MPI_INT,0,tag,MPI_COMM_WORLD);
  free(s_entries);

  int* final_splitters = calloc(P-1,sizeof(int));
  /* root processor does a sort, determinates splitters and broadcasts them */
  if (rank == 0) {
    int* splitters = calloc(P*(P-1),sizeof(int));
    int* message_in = calloc(P-1,sizeof(int));
    for (i=P-1;i>=0;i--){
      MPI_Recv(message_in,P-1,MPI_INT,i,tag,MPI_COMM_WORLD,&status);
      int j;
      for (j = 0;j<P-1;j++){
        splitters[i*(P-1)+j] = message_in[j];
      }
    }
    qsort(splitters,P*(P-1),sizeof(int),compare);

    //get the final splitters by picking the middle P-1 ints from the possible.
    // so shift all indices back by P/2.
    for (i=1;i<=P-1;i++){
      final_splitters[i-1] = splitters[i*P-1 - P/2];
    }
    free(splitters);
    free(message_in);
    /* every processor uses the obtained splitters to decide to send
     * which integers to whom */
    for (i=1;i<P;i++){
      MPI_Send(final_splitters,P-1,MPI_INT,i,tag,MPI_COMM_WORLD);
    }
  } else {
    MPI_Recv(final_splitters,P-1,MPI_INT,0,tag,MPI_COMM_WORLD,&status);
  }

  //index counter for determining to what processor to send what int.
  int counter = 0;
  //records the first int in the loop
  int prev_int = 0;
  //records the number of ints to send to each processor.
  int* sizes_out = calloc(P,sizeof(int));
  int* sizes_in = calloc(P,sizeof(int));
  for (i = 0;i<N;i++){
    if (vec[i] >= final_splitters[counter]) {
      //record the number of ints to send to the (counter)th bucket for sizes.
      sizes_out[counter] = i-prev_int;
      if (counter == P-1) {
        //set this condition to cut the looping at the final counter.
        sizes_out[P-1] = N-prev_int;
        break;
      }
      counter++;
      prev_int = i;
    }
  }

  /* send and receive: either you use MPI_AlltoallV, or
   * (and that might be easier), use an MPI_Alltoall to share
   * with every processor how many integers it should expect,
   * and then use MPI_Send and MPI_Recv to exchange the data */
  //disperses the ints to each of the other processors. 
  MPI_Alltoall(sizes_out,1,MPI_INT,sizes_in,1,MPI_INT,MPI_COMM_WORLD);

  //array for expecting the number of ints from each other processor.
  int* displ_out = calloc(P,sizeof(int));
  int* displ_in = calloc(P,sizeof(int));
  displ_out[0] = 0;
  displ_in[0] = 0;
  //compute vector sizes for partitions of vectors being sent out and in.
  for (i=1;i<P;i++) {
    displ_out[i] = sizes_out[i-1] + displ_out[i-1];
    displ_in[i] = sizes_in[i-1] + displ_in[i-1];
  }
  //compute total size to expect;
  int size_in=0;
  for (i=0;i<P;i++) {
    size_in += sizes_in[i];
  }
  int* vec_in = calloc(size_in,sizeof(int));

  //send vec out among all other procs and receive into vec_in
  MPI_Alltoallv(vec,sizes_out,displ_out,MPI_INT,vec_in,sizes_in,displ_in,MPI_INT,MPI_COMM_WORLD);

  /* local sort */
  qsort(vec_in,size_in,sizeof(int),compare);

  /* every processor writes its result to a file */
  int* final_vec = calloc(P*N,sizeof(int));
  //MPI_Gather all sizes into array of final_sizes
  int* final_sizes = calloc(P,sizeof(int));
  int* final_disp = calloc(P,sizeof(int));
  MPI_Gather(&size_in,1,MPI_INT,final_sizes,1,MPI_INT,0,MPI_COMM_WORLD);
  //Broadcast final sizes to all ranks
  MPI_Bcast(final_sizes,P,MPI_INT,0,MPI_COMM_WORLD);
  //compute displacement of final vector
  final_disp[0] = 0;
  for (i=1;i<P;i++) {
    final_disp[i] = final_disp[i-1] + final_sizes[i-1];
  }
  //MPI_Gatherv everything into one vector at root
  MPI_Gatherv(vec_in,size_in,MPI_INT,final_vec,final_sizes,final_disp,MPI_INT,0,MPI_COMM_WORLD);
  free(final_sizes);
  free(final_disp);

  if (rank == 0) {
    FILE* sorted_vec = NULL;
    char* filename = "sorted_vec.txt";
    sorted_vec = fopen(filename,"w+");
    if (NULL == sorted_vec) {
      printf("Error opening file\n");
      return -1;
    }
    //fprintf the elements of the vector for each row.
    for (i = 0;i<P*N;i++) {
      fprintf(sorted_vec,"%d\n",final_vec[i]);
    }

    fclose(sorted_vec);
  }

  free(final_splitters);
  free(vec);
  free(vec_in);
  free(sizes_out);
  free(sizes_in);
  MPI_Finalize();
  return 0;
}
