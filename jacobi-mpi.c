#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <sys/time.h>

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
  //struct timeval t1,t2;
  //dimension of matrix and vector. Matrix is square matrix
  
    //start time
    //gettimeofday(&t1,NULL);
    MPI_Status status;

      unsigned int dim,iter; //number of iterations
      if (argc == 1) {
        /*Default value for dim:
         minimum case where the start point does not equal the endpoint
         but since this is a very imprecise partition, this is not
         desirable.*/
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

      //set entries of the partition of u. Everything in between.
      int i,k;
      for (i = 0;i<divlength;i++) ui[i+1] = u[start+i];

      //MPI send and receive u 
      int dest, tag = 99;

      double prev_endpt; double next_endpt;
      double *part_in = calloc((divlength+2),sizeof(double));
      double *part_out = calloc((divlength+2),sizeof(double));
      if (rank == 0) {
        dest = nprocs == 1 ? 0:1;
	if (j > 1) {
	//receive the vector partition back from nprocs - 1 if j > 1 
          MPI_Recv(part_in,divlength+2,MPI_DOUBLE,nprocs-1,tag,MPI_COMM_WORLD,&status);
          for (i = 0;i<divlength+2;i++) ui[i] = part_in[i];
	  //sends endpoint back to nprocs - 1
          if (nprocs == 2) {
	    prev_endpt = ui[divlength];
	    MPI_Send(&prev_endpt,1,MPI_DOUBLE,1,tag,MPI_COMM_WORLD);
          } 
        } else if (j == 1) {
        prev_endpt = ui[divlength];
        MPI_Send(&prev_endpt,1,MPI_DOUBLE,dest,tag,MPI_COMM_WORLD);
        MPI_Recv(&next_endpt,1,MPI_DOUBLE,dest,tag,MPI_COMM_WORLD,&status);
        ui[divlength+1] = next_endpt;
        }
        jacobi(uin,ui,f,0,divlength,dim);

	if (nprocs == 1 && j == iter){
	  for (k = 0;k<dim;k++) printf("%lf ",uin[k+1]);
          printf("\n");
	}

	if (iter > 1) {
          for (i = 0;i<divlength+2;i++) part_out[i] = uin[i];
          MPI_Send(part_out,divlength+2,MPI_DOUBLE,nprocs-1,tag,MPI_COMM_WORLD);
        }
      } else if (rank == nprocs - 1) {
        MPI_Recv(&next_endpt,1,MPI_DOUBLE,nprocs-2,tag,MPI_COMM_WORLD,&status);
        ui[0] = next_endpt;
        //send back endpoint if first iteration.
        if (j == 1) {
	  prev_endpt = ui[1];
          MPI_Send(&prev_endpt,1,MPI_DOUBLE,nprocs-2,tag,MPI_COMM_WORLD);
	}
        for (k = nprocs-2;k>=0;k--) {
	  MPI_Recv(part_in,divlength+2,MPI_DOUBLE,k,tag,MPI_COMM_WORLD,&status);
          for (i = 0;i<divlength;i++) u[k*divlength+i] = part_in[i+1];
        }
        jacobi(uin,ui,f,start,divlength,dim);

	//update last part of vector
        for (i = 0;i<divlength;i++) u[start+i] = uin[i+1];

        //print u after every iteration to check for consistency on processors
	// to use this, simply remove the comment markers on the following 3 lines
        /*printf("Iter: %d\n",j);
        for (k = 0;k<dim;k++) printf("%lf ",u[k]);
        printf("\n");*/

	//print final vector after final iteration has been reached.
        if (j == iter) {
	  for (k = 0;k<dim;k++) printf("%lf ",u[k]);
          printf("\n");
	}

	if (j < iter) {
	//send all other pieces back to other procs if j < iter
          for (k = 0;k<nprocs-1;k++) {
            if (k == 0) for (i = 0;i<divlength+1;i++) part_in[i+1] = u[i];
	    else for (i = 0;i<divlength+2;i++) part_in[i] = u[k*divlength+i-1];
	    MPI_Send(part_in,divlength+2,MPI_DOUBLE,k,tag,MPI_COMM_WORLD);
          }
        }
      } else {
	//receive the previous computations packed into one vector for final processing.
	//receive the vector partition back from nprocs - 1 and get entries for u. 
	if (j > 1) {
	//receive the vector partition back from nprocs - 1 if j > 1 
          MPI_Recv(part_in,divlength+2,MPI_DOUBLE,nprocs-1,tag,MPI_COMM_WORLD,&status);
          for (i = 0;i<divlength+2;i++) ui[i] = part_in[i];
	  //sends endpoint back to nprocs - 1
          if (rank == nprocs - 2) {
	    prev_endpt = ui[divlength];
	    MPI_Send(&prev_endpt,1,MPI_DOUBLE,nprocs-1,tag,MPI_COMM_WORLD);
          } 
        } else if (j == 1) {
        MPI_Recv(&next_endpt,1,MPI_DOUBLE,rank-1,tag,MPI_COMM_WORLD,&status);
        ui[0] = next_endpt;
        prev_endpt = ui[1]; 
        MPI_Send(&prev_endpt,1,MPI_DOUBLE,rank-1,tag,MPI_COMM_WORLD);
        prev_endpt = ui[divlength];
        MPI_Send(&prev_endpt,1,MPI_DOUBLE,rank+1,tag,MPI_COMM_WORLD);
        MPI_Recv(&next_endpt,1,MPI_DOUBLE,rank+1,tag,MPI_COMM_WORLD,&status);
        ui[divlength+1] = next_endpt;
	}
        jacobi(uin,ui,f,start,divlength,dim);

        //part_out = uin;
        for (i = 0;i<divlength+2;i++) part_out[i] = uin[i];
        MPI_Send(part_out,divlength+2,MPI_DOUBLE,nprocs-1,tag,MPI_COMM_WORLD);
      }

      //calculations used to compute residue. Not complete in this version.
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
      free(uin);
      free(part_out);
      free(part_in);
      }

    //end time
    MPI_Finalize();

    free(u);

    //Time elapsed. To use, simply uncomment the following code
    /*if (rank == nprocs - 1) {
      gettimeofday(&t2,NULL);
      int k;
      for (k = 0;k<dim;k++) printf("%lf ",u[k]);
      printf("\n");

      printf("Time elapsed for %d procs, %d partitions: %lf sec.\n", nprocs,dim + 1,((double)(t2.tv_usec-t1.tv_usec)/1000000 + (double)(t2.tv_sec - t1.tv_sec)));
    }*/

  return 0;
}
