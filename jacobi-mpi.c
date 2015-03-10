#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <sys/time.h>

/* customized vector difference function. In this case, b is
   guaranteed to be constant.
*/

/* customized matrix-vector product to compute for specific A
   We want to solve for h^2*Au=h^2, so h^2*A has integral entries
*/

/*
void prod(double* p, double* u,unsigned int start, int divlength, unsigned int dim)
{
  //borderline cases
  if (start == 0) p[0] = 2*u[1] -u[2];
  else p[0] = -u[0] + 2*u[1] -u[2];

  if (start + divlength == dim) p[divlength-1] = -u[divlength-1] + 2*u[divlength];
  else p[divlength-1] = -u[divlength-1] + 2*u[divlength] -u[divlength+1];

  //all other columns are 0. Only three nonzero terms.
  unsigned int i;
  for (i=1;i<divlength - 1;i++)
  {
    p[i] = -u[i] + 2*u[i+1] -u[i+2];
  }
}
*/

void jacobi(double* u_new, double* u, double h2,unsigned int start,int divlength,unsigned int dim)
{
  int i;
  for (i = 0;i<divlength;i++)
  {
    double msum;
    /*Sums everything in a specific row of h^2A except for diagonal elements
      h^2A is sparse, enabling it to be reduced to a simple if statement
      for every step */ 
    if (start+i == 0) msum = -u[1];
    else if (start+i == dim -1) msum = -u[divlength-1];
    else msum= -u[i]-u[i+2];
    u_new[i+1] = 1.0/2.0*(h2-msum);
  }
}

int main(int argc,char* argv[]) {

  /* compute time elapsed. The function is taken from the following URL:
     http://stackoverflow.com/questions/5248915/execution-time-of-c-program
  */
  struct timeval t1,t2;
  //dimension of matrix and vector. Matrix is square matrix
  
    //start time
    gettimeofday(&t1,NULL);
    MPI_Status status;

      unsigned int dim,iter; //number of iterations
      if (argc == 1) {
        /*Default value for dim:
         minimum case where the start point does not equal the endpoint
         but since this is a very imprecise partition, this is highly
         undesirable.*/
        printf("No partition amount specified. Partitioning into 2 intervals.\n");
        printf("No iteration amount specified. Jacobi algorithm will iterate 10 times.\n");
        dim = 2;
        iter = 10;
      } else if (argc == 2) {
        printf("No iteration amount specified. Jacobi algorithm will iterate 10 times.\n");
        dim = atoi(argv[1]);
        iter = 10;
      } else {
        dim = atoi(argv[1]);
        iter = atoi(argv[2]);
      }

    double h = 1.0/(dim+1);
    double f = h*h; //function f is 1

    int nprocs,rank;
    unsigned int start;
    MPI_Init(&argc,&argv);

    //solve u. Use argv[2] to pick numerical algorithm.
    /*pre-computing res_0 for efficiency. Since we deal with f=1,
      res_0 = sqrt(1 + 1 + ... + 1) = sqrt(dim)
    
      in both cases, Au^0 = 0, so res_0 is just the norm of f
       For f = 1, the norm^2 would be dim, so norm(f,dim) = sqrt(dim)
    */
    //double res_0 = sqrt(dim);
    //double res = res_0; //compute the residual

    //use here to save the entries
    double *u = calloc(dim,sizeof(double));

    //Begin computation
    int j;
    for (j=1;j<=iter;j++) { 
      MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
      if (dim % nprocs != 0) {
        printf("Number of processors does not evenly divide vector length.\n");
        MPI_Finalize();
        abort();
      }
      int divlength = dim/nprocs;

      MPI_Comm_rank(MPI_COMM_WORLD,&rank);
      /*Index for start of u corresponding to a particular processor. Since we are guaranteed
        that the number of processors divides the number of partitions, we only need to record start*/
      start = rank*divlength;

      //vectors for computation. For safety of computation, all entries are initialized
      // to 0
      //double *p = calloc((divlength),sizeof(double));
      //double *d = calloc((divlength),sizeof(double));
      double *ui = calloc((divlength+2),sizeof(double));	//ith segment of u
      double *uin = calloc((divlength+2),sizeof(double));	//new ith segment of u
      double *un = calloc(dim,sizeof(double));			//new u. Records the segments so far.

      //set entries of the partition of u. Everything in between.
      int i,k;
      for (i = 0;i<divlength;i++) ui[i+1] = u[start+i];

      //MPI send and receive u 
      int destination,origin, tag = 99;

      double *prev_u = calloc(dim,sizeof(double)); double *next_u=calloc(dim,sizeof(double));
      if (start == 0) {
        destination = nprocs == 1 ? 0:1;
        origin = nprocs - 1;
        ui[divlength+1] = u[divlength];
        jacobi(uin,ui,f,0,divlength,dim);
        for (i = 0;i<divlength;i++) un[i] = uin[i+1];
        for (k = 0;k<dim;k++) prev_u[k] = un[k];
        MPI_Send(prev_u,dim,MPI_DOUBLE,destination,tag,MPI_COMM_WORLD);
	//receive immediately in case we only have one interval. Serves no other purpose.
        MPI_Recv(next_u,dim,MPI_DOUBLE,nprocs-1,tag,MPI_COMM_WORLD,&status);
        for (k = 0;k<dim;k++)u[k] = next_u[k];
      } else if (start+divlength == dim) {
        MPI_Recv(next_u,dim,MPI_DOUBLE,nprocs-2,tag,MPI_COMM_WORLD,&status);
        ui[0] = u[start - 1];
        jacobi(uin,ui,f,start,divlength,dim);
        for (i = 0;i<start;i++) un[i] = next_u[i];
        for (i = 0;i<divlength;i++) un[start+i] = uin[i+1];
        for (k = 0;k<dim;k++) u[k] = un[k];
        //PRINT DEBUG: print vector to check
        printf("Iter: %d\n",j);
        for (k = 0;k<dim;k++) printf("%lf ",u[k]);
        printf("\n");

        for (k = 0;k<dim;k++) prev_u[k] = u[k];
        MPI_Send(prev_u,dim,MPI_DOUBLE,0,tag,MPI_COMM_WORLD); //send something back to receive at rank 0
      } else {
	//receive the previous computations packed into one vector for final processing.
        MPI_Recv(next_u,dim,MPI_DOUBLE,rank-1,tag,MPI_COMM_WORLD,&status);
        ui[0] = u[start - 1];
        ui[divlength+1] = u[start+divlength];
        jacobi(uin,ui,f,start,divlength,dim);
	//PRINT DEBUG
        for (i = 0;i<divlength;i++) printf("%lf ", uin[i+1]);
	printf("\n");

        for (i = 0;i<start;i++) un[i] = next_u[i];
        for (i = 0;i<divlength;i++) un[start+i] = uin[i+1];
        for (k = 0;k<dim;k++) prev_u[k] = un[k];
        MPI_Send(prev_u,dim,MPI_DOUBLE,rank+1,tag,MPI_COMM_WORLD);
      }

      //prod(p,un,start,divlength,dim);			//p = h^2*Au
      //for(i=0;i<divlength;i++) d[i] = p[i] - f;	//d = p - h^2f
      //double message_in,sum;

      //pass messages
      /*if (rank == 0) sum = 0.0;
      for (i = 0;i<divlength;i++) sum += d[i]*d[i];

      if (rank == nprocs - 1) {
        res = (dim+1)*(dim+1)*sqrt(sum);
        printf("Res: %lf\n",res); //print residual to check for convergence.
      }
      */
      //free(p);
      //free(d);
      free(ui);
      free(un);
      free(prev_u);
      free(next_u);
      }

    //end time
    MPI_Finalize();

    free(u);
    if (rank == nprocs - 1) {
      gettimeofday(&t2,NULL);

      printf("Time elapsed: %lf sec.\n", ((double)(t2.tv_usec-t1.tv_usec)/1000000 + (double)(t2.tv_sec - t1.tv_sec)));
    }

  return 0;
}
