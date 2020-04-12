#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <string>
#include <cassert>
#include <vector>
#include <iostream>
#include <armadillo>
#include <utility>
#include <omp.h>
#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/openmp/openmp.hpp>
#include <boost/random.hpp>

typedef std::vector< double > state_type;

__device__ double atomicAddD(double* address, double val)
{
    unsigned long long int* address_as_ull =
                             (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__
void calcproperties(double *flux_aux_d,double *x_vec_lin_d, double *A_lin_d, double *G_lin_d, int N, int steps)
{
 	int i = blockIdx.x*blockDim.x + threadIdx.x;

  if (i < steps) 
  	{
		for (int place = 0; place < N; ++place)
		{
			/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			for (int l = 0; l < N; ++l)
			{
				atomicAddD(&flux_aux_d[place+N*l],1000.0*A_lin_d[l+N*place]*sin(x_vec_lin_d[0+i*2+steps*2*l]-x_vec_lin_d[0+i*2+steps*2*place])*x_vec_lin_d[1+i*2+steps*2*place]/N);
			}
			atomicAddD(&flux_aux_d[place+N*N],-1000.0*G_lin_d[place]*(x_vec_lin_d[1+i*2+steps*2*place]*x_vec_lin_d[1+i*2+steps*2*place]));
			/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		}
	}
}


void inicialcond(state_type &x,int N,boost::mt19937 &rng,int caso)
{
    boost::uniform_real<> unif( 0, 2*M_PI );//la distribucion de probabilidad uniforme entre cero y 2pi
    boost::variate_generator< boost::mt19937&, boost::uniform_real<> > gen( rng , unif );//gen es una funcion que toma el engine y la distribucion y devuelve el numero random

    if(caso==0)
    {
    	FILE *w= fopen("Xi.txt", "w");
    	for (int i = 0; i < N; ++i)
		{
			fprintf(w, "%f  ",gen() );
			fprintf(w, "%f\n",0.0 );
		}
		fclose(w);
		FILE *r= fopen("Xi.txt", "r");
		for (int i = 0; i < N; ++i)
		{
			fscanf(r, "%lf", &x[2*i]); // posicion inicial i
			fscanf(r, "%lf", &x[2*i+1]); // momento inicial i
		}
		fclose(r);
    }
    if(caso==1)
    {
    	FILE *r= fopen("Xi.txt", "r");
		for (int i = 0; i < N; ++i)
		{
			fscanf(r, "%lf", &x[2*i]); // posicion inicial i
			fscanf(r, "%lf", &x[2*i+1]); // momento inicial i
		}
		fclose(r);
    }
}

void fillA(arma::Mat<double> &A,int N,boost::mt19937 &rng,int caso)
{

    boost::uniform_real<> unif( 0, 1 );//la distribucion de probabilidad uniforme entre cero y 2pi
    boost::variate_generator< boost::mt19937&, boost::uniform_real<> > gen( rng , unif );//gen es una funcion que toma el engine y la distribucion y devuelve el numero random
    double prob_0=0;
    if (caso==0)
    {
    	FILE *w= fopen("Ai.txt", "w");
    	for (int i = 0; i < N; ++i)
		{
			for (int j = 0; j <= i; ++j)
			{
				if(gen()>=prob_0)
				{
					fprintf(w, "%f  ",1.0);
				}
				else
				{
					fprintf(w, "%f  ",0.0);
				}

			}
		}
		fclose(w);
		FILE *r= fopen("Ai.txt", "r");
    	for (int i = 0; i < N; ++i)
		{
			for (int j = 0; j <= i; ++j)
			{
				fscanf(r,"%lf",&A(i,j));
			}
		}
		fclose(r);
    	for (int i = 0; i < N; ++i)
		{
			for (int j = N-1; j > i; --j)
			{
				A(i,j)=A(j,i);
			}
		}
    }
    if(caso==1)
    {
    	FILE *r= fopen("Ai.txt", "r");
    	for (int i = 0; i < N; ++i)
		{
			for (int j = 0; j <= i; ++j)
			{
				fscanf(r,"%lf",&A(i,j));
			}
		}		
		fclose(r);
    	for (int i = 0; i < N; ++i)
		{
			for (int j = N-1; j > i; --j)
			{
				A(i,j)=A(j,i);
			}
		}
    }
}

void fillG(std::vector<double> &G,int N,boost::mt19937 &rng,int caso)
{

    boost::normal_distribution<> unif(2.5, 0.2 );//la distribucion de probabilidad uniforme entre cero y 2pi
    boost::variate_generator< boost::mt19937&, boost::normal_distribution<> > gen( rng , unif );//gen es una funcion que toma el engine y la distribucion y devuelve el numero random

    if(caso==0)
    {
    	FILE *w= fopen("Gi.txt", "w");
		for (int i = 0; i < N; ++i)
		{
			fprintf(w, "%lf  ", gen());
		}
		fclose(w);
		FILE *r= fopen("Gi.txt", "r");
		for (int i = 0; i < N; ++i)
		{
			fscanf(r, "%lf", &G[i]);
		}
		fclose(r);
	}
	if(caso==1)
	{
		FILE *r= fopen("Gi.txt", "r");
		for (int i = 0; i < N; ++i)
		{
			fscanf(r, "%lf", &G[i]);
		}
		fclose(r);
	}
}

void fillI(std::vector<double> &I,int N,boost::mt19937 &rng,int caso)
{
    if (caso==0)
    {
    	FILE *w= fopen("Ii.txt", "w");
		for (int i = 0; i < N; ++i)
		{
			fprintf(w, "%lf  ", 1.0);
		}
		fclose(w);
		FILE *r= fopen("Ii.txt", "r");
		for (int i = 0; i < N; ++i)
		{
			fscanf(r, "%lf", &I[i]);
		}
		fclose(r);
    }
    if(caso==1)
    {
		FILE *r= fopen("Ii.txt", "r");
		for (int i = 0; i < N; ++i)
		{
			fscanf(r, "%lf", &I[i]);
		}
		fclose(r);
    }

}

void fillW(std::vector<double> &Fw,int N,boost::mt19937 &rng,int caso)
{
    boost::uniform_real<> unif( 0, 10 );//la distribucion de probabilidad uniforme entre cero y 2pi
    boost::variate_generator< boost::mt19937&, boost::uniform_real<> > gen( rng , unif );//gen es una funcion que toma el engine y la distribucion y devuelve el numero random

    if(caso==0)
    {
    	FILE *w= fopen("Wi.txt", "w");
    	for (int i = 0; i < N; ++i)
		{
			fprintf(w, "%lf  ", 1.0);
		}
		fclose(w);
		FILE *r= fopen("Wi.txt", "r");
    	for (int i = 0; i < N; ++i)
		{
			fscanf(r, "%lf", &Fw[i]);
		}
		fclose(r);
    }
    if(caso==1)
    {
		FILE *r= fopen("Wi.txt", "r");
    	for (int i = 0; i < N; ++i)
		{
			fscanf(r, "%lf", &Fw[i]);
		}
		fclose(r);
    }
}

void fillFw(std::vector<double> &Fw,int N,boost::mt19937 &rng,int caso)
{
    boost::uniform_real<> unif( 0, 10 );//la distribucion de probabilidad uniforme entre cero y 2pi
    boost::variate_generator< boost::mt19937&, boost::uniform_real<> > gen( rng , unif );//gen es una funcion que toma el engine y la distribucion y devuelve el numero random

    if(caso==0)
    {
    	FILE *w= fopen("Fwi.txt", "w");
    	for (int i = 0; i < N; ++i)
		{
			if(i==0)
			{
				fprintf(w, "%lf  ", 10.0);
			}
			else
			{
				fprintf(w, "%lf  ", 0.0);
			}
			
		}
		fclose(w);
		FILE *r= fopen("Fwi.txt", "r");
    	for (int i = 0; i < N; ++i)
		{
			fscanf(r, "%lf", &Fw[i]);
		}
		fclose(r);
    }
    if(caso==1)
    {
		FILE *r= fopen("Fwi.txt", "r");
    	for (int i = 0; i < N; ++i)
		{
			fscanf(r, "%lf", &Fw[i]);
		}
		fclose(r);
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double calcTn(double P, double I)
{
	double ecin=0;
	ecin=0.5*(P*P/I);
	return ecin;
}

double calcT(std::vector< state_type > x_vec,std::vector<double> I,int N,int time)
{
	double ecin=0;
	for (int i = 0; i < N; ++i)
	{
		ecin=ecin+calcTn(x_vec[time][2*i+1],I[i]);
	}
	return ecin;
}

double calcEpotn(arma::Mat<double> A, std::vector< state_type > x_vec,int N,double K,int place,int time)
{
	double epot=0;
	for (int i = 0; i < N; ++i)
	{
		epot=epot+A(place,i)*cos(x_vec[time][2*i]-x_vec[time][2*place]);
	}
	epot=epot*K/(2*N);
	return -epot;
}

double calcEpot(arma::Mat<double> A, std::vector< state_type > x_vec, int N, double K,int time)
{
	double epot=0;
	for (int i = 0; i < N; ++i)
	{
		epot=epot+calcEpotn(A,x_vec,N,K,i,time);
	}
	return epot;
}

double calcH(arma::Mat<double> A,std::vector< state_type > x_vec, std::vector<double> I, int N, double K,int time)
{
	double H=0;
	H=calcT(x_vec,I,N,time)+calcEpot(A,x_vec,N,K,time);
	return H;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void calculateStuff(arma::Mat<double> A,int steps, double *x_vec_lin,int N,std::vector<double> G,double *flux_aux_d) //1 tiempo. 2 posicion. 3 momento. 4 energia potencial. 5 energia cinetica. 6 energia. 7 energia Total
{
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// declara
	double *x_vec_lin_d;
	double *A_lin;
	double *A_lin_d;
	double *G_lin;
	double *G_lin_d;
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// aloca
	A_lin=(double*)malloc(sizeof(double)*N*N);
	G_lin=(double*)malloc(sizeof(double)*N);

	if(cudaMalloc(&x_vec_lin_d, sizeof(double)*2*N*steps)!=cudaSuccess)
	{
		printf("erroralocaxvec\n");
		return;
	}
	if(cudaMalloc(&A_lin_d, sizeof(double)*N*N)!=cudaSuccess)
	{
		printf("erroralocaA\n");
		return;
	}
	if(cudaMalloc(&G_lin_d, sizeof(double)*N)!=cudaSuccess)
	{
		printf("erroralocaG\n");
		return;
	}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// evalua
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			A_lin[j+N*i]=A(j,i);
		}
		G_lin[i]=G[i];
	}
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// copia


	if(cudaMemcpy(x_vec_lin_d, x_vec_lin, 2*steps*N*sizeof(double), cudaMemcpyHostToDevice)!=cudaSuccess)
	{
		printf("xvecmal\n");
		return;
	}
	if(cudaMemcpy(A_lin_d, A_lin, N*N*sizeof(double), cudaMemcpyHostToDevice)!=cudaSuccess)
	{
		printf("Amal\n");
		return;
	}
	if(cudaMemcpy(G_lin_d, G_lin, N*sizeof(double), cudaMemcpyHostToDevice)!=cudaSuccess)
	{
		printf("Gmal\n");
		return;
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



	calcproperties<<<steps/256+1,256>>>(flux_aux_d,x_vec_lin_d,A_lin_d,G_lin_d,N,steps);

	cudaFree(x_vec_lin_d);
	cudaFree(A_lin_d);
	cudaFree(G_lin_d);

	free(A_lin);
	free(G_lin);
}

double calcdt(int N)
{
	double dt;
	FILE *f=fopen("save.txt","r");
	fscanf(f, "%lf", &dt);
	for (int i = 0; i < N; ++i)
	{
		fscanf(f, "%lf", &dt); // posicion inicial i
		fscanf(f, "%lf", &dt); // momento inicial i
	}
	fscanf(f, "%lf", &dt);
	return(dt);
}

void itera(arma::Mat<double> &A,std::vector<double> &G,int N,double T_t, double StartPoint)
{
	double FinishPoint=1-StartPoint;
	double T_t_i=T_t;
	//T_t=T_t*FinishPoint;
	double dt=calcdt(N);
	printf("%lf\n",dt );

	int pasos=1+(int)((double)(2.0*N+N*N+N*(N+1.0)+2.0*N*T_t/dt)/70000000.0);
	printf("%d\n",pasos );

	int steps=T_t/(dt*pasos);
	printf("%d\n", steps );

//////////////////////////////////////////////////////////////////////////////////////////////////
	double *flux_aux;
	double *flux_aux_d;
	double *x_vec_lin;
	///
	flux_aux=(double*)malloc(sizeof(double)*N*(N+1));
	if(cudaMalloc(&flux_aux_d, sizeof(double)*N*(N+1))!=cudaSuccess)
	{
		printf("erroralocaflux\n");
		return;
	}
	x_vec_lin=(double*)malloc(sizeof(double)*2*N*steps);
	///
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N+1; ++j)
		{
			flux_aux[i+N*j]=0;
		}
	}
	if(cudaMemcpy(flux_aux_d, flux_aux, (N+1)*N*sizeof(double), cudaMemcpyHostToDevice)!=cudaSuccess)
	{
		printf("fluxmal\n");
		return;
	}	
//////////////////////////////////////////////////////////////////////////////////////////////////
    double aux;
	FILE *f=fopen("save.txt","r");
	for (int k = 0; k < StartPoint*T_t_i/dt; ++k)
	{
		fscanf(f, "%lf", &aux);
		for (int i = 0; i < N; ++i)
		{
			fscanf(f, "%lf", &aux);
			fscanf(f, "%lf", &aux);
		}
	}
    for (int k = 0; k < pasos; ++k)
    {
		for (int j = 0; j < steps; ++j)
		{
			if(j%(steps/100)==0 && j < steps)
			{
				printf("scanning savestate: %d  (%d/%d)\n", (int)(100.0*j/steps),k+1,pasos);
			}
			fscanf(f, "%lf", &aux);
			for (int i = 0; i < N; ++i)
			{
				fscanf(f, "%lf", &x_vec_lin[0+j*2+steps*2*i]);
				fscanf(f, "%lf", &x_vec_lin[1+j*2+steps*2*i]);
			}
		}
		printf("cudacall (%d/%d)\n",k+1,pasos);
		calculateStuff(A,steps,x_vec_lin,N,G,flux_aux_d);

    }
	
	if(cudaMemcpy(flux_aux, flux_aux_d, (N+1)*N*sizeof(double), cudaMemcpyDeviceToHost)!=cudaSuccess)
	{
		printf("fluxmal\n");
		return;
	}

	FILE *g=fopen("T.txt","w");
	for( int i=0; i<N; ++i )
	{
		if(i%(N/100)==0 && i < N)
		{
			printf("printing: %d \n", (int)(100.0*i/N));
		}
		for (int j = 0; j < N+1; ++j)
		{
			fprintf(g,"%.15lf	  ",flux_aux[i+N*j]/(T_t/dt)); //1 posicion. 2 momento. 3 energia potencial. 4 energia cinetica. 5 energia total
		}
		if(flux_aux[i+N*N]>0)
		{
			printf("%d tiene su gamma alreves\n", i);
		}
		fprintf(g, "\n");
	}
}

int main()
{
    using namespace std;
    using namespace boost::numeric::odeint;

    boost::mt19937 rng(static_cast<unsigned int>(std::time(0)));  /// el engine para generar numeros random

///////////////////////////////////////////////////////////////////////
    int N;
    printf("N: ");
    std::cin >>N;

    double T_t;
    printf("Total Time: ");
    std::cin >>T_t;

    double StartPoint;
    printf("Start Point [x:1]: ");
    std::cin >>StartPoint;

    arma::Mat<double> A(N,N);
    vector<double> G(N);

//////////////////////////////////////////////////////////////////////////
	fillA(A,N,rng,1);
	fillG(G,N,rng,1);
////////////////////////////////////////////////////////////////////////////
	itera(A,G,N,T_t,StartPoint);
	printf("cuda N=%d\n",N);
	printf("factor 1000!\n");

	return 0;
}