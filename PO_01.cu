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

__global__
void calcproperties(double *ecin_aux_d,double *epot_aux_d, double *flux1_aux_d, double *flux_aux_d, double *elost_aux_d, double *x_vec_lin_d, double *I_lin_d, double *A_lin_d, double *G_lin_d, int N, int steps,double K, double *xmed_aux_d)
{
 	int i = blockIdx.x*blockDim.x + threadIdx.x;
	double epot=0;
	double f=0;
	double flux=0;
    double cos_sum = 0.0 , sin_sum = 0.0;


  if (i < steps) 
  	{
		for (int place = 0; place < N; ++place)
		{
			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			ecin_aux_d[place+i*N]=0.5*x_vec_lin_d[1+i*2+steps*2*place]*x_vec_lin_d[1+i*2+steps*2*place]/I_lin_d[place];
			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			epot=0;
			for (int l = 0; l < N; ++l)
			{
				epot=epot-A_lin_d[l+N*place]*cos(x_vec_lin_d[0+i*2+steps*2*l]-x_vec_lin_d[0+i*2+steps*2*place]);
			}
			epot=epot*K/(2*N);
			epot_aux_d[place+i*N]=epot;
			/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			f=0;
			flux=0;
			for (int l = 1; l < N; ++l)
			{
				flux=A_lin_d[l+N*place]*sin(x_vec_lin_d[0+i*2+steps*2*l]-x_vec_lin_d[0+i*2+steps*2*place])*x_vec_lin_d[1+i*2+steps*2*place];		
				f=f+flux;
			}
			f=f*(K/N);
			flux_aux_d[place+i*N]=f;
			flux1_aux_d[place+i*N]=(K/(N))*A_lin_d[0+N*place]*sin(x_vec_lin_d[0+i*2+steps*2*0]-x_vec_lin_d[0+i*2+steps*2*place])*x_vec_lin_d[1+i*2+steps*2*place];
			/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

			elost_aux_d[place+i*N]=-G_lin_d[place]*(x_vec_lin_d[1+i*2+steps*2*place]*x_vec_lin_d[1+i*2+steps*2*place]); //bien

			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		}
    	cos_sum = 0.0 , sin_sum = 0.0;
    	for( size_t l=1 ; l<N ; ++l )
    	{
        	cos_sum += cos( x_vec_lin_d[0+i*2+steps*2*l] );
       	 	sin_sum += sin( x_vec_lin_d[0+i*2+steps*2*l] );
    	}
    	cos_sum /= double( N-1 );
    	sin_sum /= double( N-1 );
		xmed_aux_d[i]=atan2( sin_sum , cos_sum );		
	}
}

class harm_osc 
{
    double m_K;
    int m_N;
    std::vector< double >& m_I;
    std::vector< double >& m_F;
    arma::Mat<double> &m_A;
    std::vector< double >& m_G;
    std::vector< double >& m_Fw;

	public:
    harm_osc( double K , int N, std::vector< double > &I,arma::Mat<double> &A,std::vector< double > &F,std::vector< double > &G,std::vector< double > &Fw) : m_K(K) , m_N(N) , m_I(I), m_A(A), m_F(F) , m_G(G), m_Fw(Fw){ }

    void operator() ( const state_type &x , state_type &dxdt , const double t  )
    {
    	double sum=0;
    	if(ceilf(t)==t && (int) t%10==0)
    	{
    		printf("tiempo: %lf\n",t);
    	}
        #pragma omp parallel for schedule(runtime) private(sum)
        for (int i = 0; i < m_N; ++i)
        {
        	sum=0;
        	for (int j = 0; j < m_N; ++j)
    		{
    			sum=sum+m_A(i,j)*sin(x[2*j]-x[2*i]);
    		}
    		sum=sum*m_K/m_N;
        	dxdt[2*i]=x[2*i+1];
        	if(t<=40000)
        	{
        		dxdt[2*i+1]= sum/m_I[i]+m_F[i]*sin(m_Fw[i]*t-x[2*i])/m_I[i]-(m_G[i]/m_I[i])*x[2*i+1];
        	}
        	else
        	{
        		dxdt[2*i+1]= sum/m_I[i]-(m_G[i]/m_I[i])*x[2*i+1];
        	}
        	
        }
    }
};

struct push_back_state_and_time
{
    std::vector< state_type >& m_states;
    std::vector< double >& m_times;

    push_back_state_and_time( std::vector< state_type > &states , std::vector< double > &times ) : m_states( states ) , m_times( times ) { }

    void operator()( const state_type &x , double t )
    {
        m_states.push_back( x );
        m_times.push_back( t );
    }
};



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

void fillA(arma::Mat<double> &A,int N,boost::mt19937 &rng,int caso,double prob_0)
{

    boost::uniform_real<> unif( 0, 1 );//la distribucion de probabilidad uniforme entre cero y 2pi
    boost::variate_generator< boost::mt19937&, boost::uniform_real<> > gen( rng , unif );//gen es una funcion que toma el engine y la distribucion y devuelve el numero random
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

void printstuff(arma::Mat<double> A,size_t steps, std::vector< state_type > &x_vec,std::vector<double> &times, std::vector<double> I,int N,double K,std::vector<double> G) //1 tiempo. 2 posicion. 3 momento. 4 energia potencial. 5 energia cinetica. 6 energia. 7 energia Total
{
	double *ecin_aux;
	double *epot_aux;
	double *flux_aux;
	double *flux1_aux;
	double *elost_aux;
	double *xmed_aux;
	double *ecin_aux_d;
	double *epot_aux_d;
	double *flux_aux_d;
	double *flux1_aux_d;
	double *elost_aux_d;
	double *xmed_aux_d;

	double *x_vec_lin;
	double *x_vec_lin_d;
	double *I_lin;
	double *I_lin_d;
	double *A_lin;
	double *A_lin_d;
	double *G_lin;
	double *G_lin_d;
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	ecin_aux=(double*)malloc(sizeof(double)*(N)*steps);
	epot_aux=(double*)malloc(sizeof(double)*(N)*steps);
	flux_aux=(double*)malloc(sizeof(double)*(N)*steps);
	flux1_aux=(double*)malloc(sizeof(double)*(N)*steps);
	elost_aux=(double*)malloc(sizeof(double)*(N)*steps);
	xmed_aux=(double*)malloc(sizeof(double)*steps);

	x_vec_lin=(double*)malloc(sizeof(double)*2*N*steps);
	I_lin=(double*)malloc(sizeof(double)*N);
	A_lin=(double*)malloc(sizeof(double)*N*N);
	G_lin=(double*)malloc(sizeof(double)*N);

	if(cudaMalloc(&ecin_aux_d, sizeof(double)*N*steps)!=cudaSuccess)
	{
		printf("erroralocaecin\n");
		return;
	}	
	if(cudaMalloc(&epot_aux_d, sizeof(double)*N*steps)!=cudaSuccess)
	{
		printf("erroralocaepot\n");
		return;
	}	
	if(cudaMalloc(&flux_aux_d, sizeof(double)*N*steps)!=cudaSuccess)
	{
		printf("erroralocaflux\n");
		return;
	}	
	if(cudaMalloc(&flux1_aux_d, sizeof(double)*N*steps)!=cudaSuccess)
	{
		printf("erroralocaflux1\n");
		return;
	}
	if(cudaMalloc(&elost_aux_d, sizeof(double)*N*steps)!=cudaSuccess)
	{
		printf("erroralocaelost\n");
		return;
	}	
	if(cudaMalloc(&xmed_aux_d, sizeof(double)*steps)!=cudaSuccess)
	{
		printf("erroralocaxmed\n");
		return;
	}	
	if(cudaMalloc(&x_vec_lin_d, sizeof(double)*2*N*steps)!=cudaSuccess)
	{
		printf("erroralocaxvec\n");
		return;
	}
	cudaMalloc(&I_lin_d, sizeof(double)*N);
	cudaMalloc(&A_lin_d, sizeof(double)*N*N);
	cudaMalloc(&G_lin_d, sizeof(double)*N);

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < steps; ++j)
		{
			for (int k = 0; k < 2; ++k)
			{
				x_vec_lin[k+j*2+steps*2*i]=x_vec[j][2*i+k];
			}
		}
		I_lin[i]=I[i];
		for (int j = 0; j < N; ++j)
		{
			A_lin[j+N*i]=A(j,i);
		}
		G_lin[i]=G[i];
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	if(cudaMemcpy(ecin_aux_d, ecin_aux,steps*N*sizeof(double), cudaMemcpyHostToDevice)!=cudaSuccess)
	{
		printf("ecinmal\n");
		return;
	}	
	if(cudaMemcpy(epot_aux_d, epot_aux, steps*N*sizeof(double), cudaMemcpyHostToDevice)!=cudaSuccess)
	{
		printf("epotmal\n");
		return;
	}	
	if(cudaMemcpy(flux_aux_d, flux_aux, steps*N*sizeof(double), cudaMemcpyHostToDevice)!=cudaSuccess)
	{
		printf("fluxmal\n");
		return;
	}
	if(cudaMemcpy(flux1_aux_d, flux1_aux, steps*N*sizeof(double), cudaMemcpyHostToDevice)!=cudaSuccess)
	{
		printf("flux1mal\n");
		return;
	}
	if(cudaMemcpy(elost_aux_d, elost_aux, steps*N*sizeof(double), cudaMemcpyHostToDevice)!=cudaSuccess)
	{
		printf("elosmal\n");
		return;
	}
	if(cudaMemcpy(xmed_aux_d, xmed_aux, steps*sizeof(double), cudaMemcpyHostToDevice)!=cudaSuccess)
	{
		printf("xmedmal\n");
		return;
	}
	if(cudaMemcpy(x_vec_lin_d, x_vec_lin, 2*steps*N*sizeof(double), cudaMemcpyHostToDevice)!=cudaSuccess)
	{
		printf("xvecmal\n");
		return;
	}
	if(cudaMemcpy(I_lin_d, I_lin, N*sizeof(double), cudaMemcpyHostToDevice)!=cudaSuccess)
	{
		printf("Imal\n");
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

	FILE *f1=fopen("ac_1.txt","w");
	FILE *f2=fopen("ac_2.txt","w");
	FILE *f3=fopen("ac_3.txt","w");
	FILE *f4=fopen("ac_4.txt","w");

	calcproperties<<<steps/256+1,256>>>(ecin_aux_d,epot_aux_d,flux1_aux_d,flux_aux_d,elost_aux_d,x_vec_lin_d,I_lin_d,A_lin_d,G_lin_d,N,steps,K,xmed_aux_d);

	if(cudaMemcpy(ecin_aux, ecin_aux_d, steps*N*sizeof(double), cudaMemcpyDeviceToHost)!=cudaSuccess)
	{
		printf("ecinmal\n");
		return;
	}
	if(cudaMemcpy(epot_aux, epot_aux_d, steps*N*sizeof(double), cudaMemcpyDeviceToHost)!=cudaSuccess)
	{
		printf("epotmal\n");
		return;
	}
	if(cudaMemcpy(flux_aux, flux_aux_d, steps*N*sizeof(double), cudaMemcpyDeviceToHost)!=cudaSuccess)
	{
		printf("fluxmal\n");
		return;
	}
	if(cudaMemcpy(flux1_aux, flux1_aux_d, steps*N*sizeof(double), cudaMemcpyDeviceToHost)!=cudaSuccess)
	{
		printf("fluxmal\n");
		return;
	}
	if(cudaMemcpy(elost_aux, elost_aux_d, steps*N*sizeof(double), cudaMemcpyDeviceToHost)!=cudaSuccess)
	{
		printf("elosmal\n");
		return;
	}
	if(cudaMemcpy(xmed_aux, xmed_aux_d, steps*sizeof(double), cudaMemcpyDeviceToHost)!=cudaSuccess)
	{
		printf("xmedmal\n");
		return;
	}

	cudaFree(ecin_aux_d);
	cudaFree(epot_aux_d);
	cudaFree(flux_aux_d);
	cudaFree(flux1_aux_d);
	cudaFree(elost_aux_d);
	cudaFree(xmed_aux_d);

	cudaFree(x_vec_lin_d);
	cudaFree(I_lin_d);
	cudaFree(A_lin_d);
	cudaFree(G_lin_d);

	free(x_vec_lin);
	free(I_lin);
	free(A_lin);
	free(G_lin);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	/*#pragma omp parallel
	#pragma omp for*/
	for( size_t i=0; i<steps; ++i )
	{
		if(i>=0 && i<steps/4)
		{
			if(i%(steps/400)==0 && i < steps/4)
			{
				printf("printing: %d \n", (int)(400.0*i/steps));
			}
			fprintf(f1,"%lf  ",times[i] );
			for (int j = 0; j < N; ++j)
			{
				fprintf(f1,"%lf	  %lf   %lf   %lf   %.10lf   %.10lf   %.10lf   %.10lf   ",x_vec[i][2*j],x_vec[i][2*j+1],epot_aux[j+i*N],ecin_aux[j+i*N],epot_aux[j+i*N]+ecin_aux[j+i*N],flux1_aux[j+i*N],flux_aux[j+i*N],elost_aux[j+i*N]); //1 posicion. 2 momento. 3 energia potencial. 4 energia cinetica. 5 energia total
			}
			fprintf(f1,"%lf   \n",xmed_aux[i]);
		}
		if(i>=steps/4 && i<steps*2/4)
		{
			fprintf(f2,"%lf  ",times[i] );
			for (int j = 0; j < N; ++j)
			{
				fprintf(f2,"%lf	  %lf   %lf   %lf   %.10lf   %.10lf   %.10lf   %.10lf   ",x_vec[i][2*j],x_vec[i][2*j+1],epot_aux[j+i*N],ecin_aux[j+i*N],epot_aux[j+i*N]+ecin_aux[j+i*N],flux1_aux[j+i*N],flux_aux[j+i*N],elost_aux[j+i*N]); //1 posicion. 2 momento. 3 energia potencial. 4 energia cinetica. 5 energia total
			}
			fprintf(f2,"%lf   \n",xmed_aux[i]);
		}
		if(i>=steps*2/4 && i<steps*3/4)
		{	
			fprintf(f3,"%lf  ",times[i] );
			for (int j = 0; j < N; ++j)
			{
				fprintf(f3,"%lf	  %lf   %lf   %lf   %.10lf   %.10lf   %.10lf   %.10lf   ",x_vec[i][2*j],x_vec[i][2*j+1],epot_aux[j+i*N],ecin_aux[j+i*N],epot_aux[j+i*N]+ecin_aux[j+i*N],flux1_aux[j+i*N],flux_aux[j+i*N],elost_aux[j+i*N]); //1 posicion. 2 momento. 3 energia potencial. 4 energia cinetica. 5 energia total
			}
			fprintf(f3,"%lf   \n",xmed_aux[i]);
		}
		if(i>=steps*3/4)
		{
			fprintf(f4,"%lf  ",times[i] );
			for (int j = 0; j < N; ++j)
			{
				fprintf(f4,"%lf	  %lf   %lf   %lf   %.10lf   %.10lf   %.10lf   %.10lf   ",x_vec[i][2*j],x_vec[i][2*j+1],epot_aux[j+i*N],ecin_aux[j+i*N],epot_aux[j+i*N]+ecin_aux[j+i*N],flux1_aux[j+i*N],flux_aux[j+i*N],elost_aux[j+i*N]); //1 posicion. 2 momento. 3 energia potencial. 4 energia cinetica. 5 energia total
			}
			fprintf(f4,"%lf   \n",xmed_aux[i]);
		}
	}	
	fclose(f1);
	fclose(f2);
	fclose(f3);
	fclose(f4);
}

void printsave(size_t steps, std::vector< state_type > &x_vec,std::vector<double> &times,int N) //1 tiempo. 2 posicion. 3 momento. 4 energia potencial. 5 energia cinetica. 6 energia. 7 energia Total
{

	FILE *f1=fopen("save_1.txt","w");
	FILE *f2=fopen("save_2.txt","w");
	FILE *f3=fopen("save_3.txt","w");
	FILE *f4=fopen("save_4.txt","w");

	/*#pragma omp parallel
	#pragma omp for*/
	for( size_t i=0; i<steps; ++i )
	{
		if(i>=0 && i<steps/4)
		{
			if(i%(steps/400)==0 && i < steps/4)
			{
				printf("printing savestate: %d \n", (int)(400.0*i/steps));
			}
			fprintf(f1,"%lf  ",times[i] );
			for (int j = 0; j < N; ++j)
			{
				fprintf(f1,"%lf	  %lf   ",x_vec[i][2*j],x_vec[i][2*j+1]); //1 posicion. 2 momento. 3 energia potencial. 4 energia cinetica. 5 energia total
			}
			fprintf(f1,"\n");
		}
		if(i>=steps/4 && i<steps*2/4)
		{
			fprintf(f2,"%lf  ",times[i] );
			for (int j = 0; j < N; ++j)
			{
				fprintf(f2,"%lf   %lf   ",x_vec[i][2*j],x_vec[i][2*j+1]); //1 posicion. 2 momento. 3 energia potencial. 4 energia cinetica. 5 energia total
			}
			fprintf(f2,"\n");
		}
		if(i>=steps*2/4 && i<steps*3/4)
		{	
			fprintf(f3,"%lf  ",times[i] );
			for (int j = 0; j < N; ++j)
			{
				fprintf(f3,"%lf	  %lf   ",x_vec[i][2*j],x_vec[i][2*j+1]); //1 posicion. 2 momento. 3 energia potencial. 4 energia cinetica. 5 energia total
			}
			fprintf(f3,"\n");
		}
		if(i>=steps*3/4)
		{
			fprintf(f4,"%lf  ",times[i] );
			for (int j = 0; j < N; ++j)
			{
				fprintf(f4,"%lf   %lf   ",x_vec[i][2*j],x_vec[i][2*j+1]); //1 posicion. 2 momento. 3 energia potencial. 4 energia cinetica. 5 energia total
			}
			fprintf(f4,"\n");
		}
	}	
	fclose(f1);
	fclose(f2);
	fclose(f3);
	fclose(f4);
}

void makeanim(std::vector< state_type > &x_vec,size_t steps,int N,FILE *gnuplotpipe)
{
    for (size_t i = 0; i < steps; ++i)
    {
        fprintf(gnuplotpipe,"plot '-' pt 7 title\"\"\n");
    	for (int j = 0; j < N; ++j)
    	{
    		fprintf(gnuplotpipe,"%lf  %lf\n",9*cos(x_vec[i][2*j]),9*sin(x_vec[i][2*j]));
    	}     
    	fprintf(gnuplotpipe,"e\n");   
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

    int load;
    printf("Load CI (0 NO, 1 YES): ");
    std::cin >>load;

    double prob_0;
    printf("Prob 0: ");
    std::cin >>prob_0;

	double K=1;
    arma::Mat<double> A(N,N);
    vector<double> I(N);
    vector<double> G(N);
    vector<double> F(N);
    vector<double> Fw(N);
	state_type x(2*N); //condiciones iniciales
    vector<state_type> x_vec;
    vector<double> times;
//////////////////////////////////////////////////////////////////////////
	fillA(A,N,rng,load,prob_0);
	fillG(G,N,rng,load);
	fillI(I,N,rng,load);
	fillFw(F,N,rng,load);
	fillW(Fw,N,rng,load);
	inicialcond(x,N,rng,load);
////////////////////////////////////////////////////////////////////////////
    harm_osc ho(K,N,I,A,F,G,Fw);
    runge_kutta4<
                      state_type , double ,
                      state_type , double ,
                      openmp_range_algebra
                    > stepper;
    int chunk_size = N/omp_get_max_threads();
    omp_set_schedule( omp_sched_static , chunk_size );
    printf("solving..\n");
	size_t steps = integrate_adaptive(stepper, ho, x , 0.0 , 500.0 , 0.01,push_back_state_and_time( x_vec , times )); //1 funcion. 2 condiciones iniciales. 3 tiempo inicial. 4 tiempo final. 5 dt inicial. 6 vector de posicion y tiempo
////////////////////////////////////////////////////////////////////////////////

	printsave(steps,x_vec,times,N);
	system("cat save_1.txt save_2.txt save_3.txt save_4.txt> save.txt");
	printstuff(A,steps,x_vec,times,I,N,K,G);
	system("cat ac_1.txt ac_2.txt ac_3.txt ac_4.txt> ac.txt");
	system("rm -f {ac_1,ac_2,ac_3,ac_4}.txt");
	system("rm -f {save_1,save_2,save_3,save_4}.txt");
	printf("N=%d\n",N);

	return 0;
}