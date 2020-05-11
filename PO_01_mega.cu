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
#include <algorithm>
#include <random>
#include <sys/resource.h>
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
				atomicAddD(&flux_aux_d[place+N*l],A_lin_d[l+N*place]*sin(x_vec_lin_d[0+i*2+steps*2*l]-x_vec_lin_d[0+i*2+steps*2*place])*x_vec_lin_d[1+i*2+steps*2*place]/N);
			}
			atomicAddD(&flux_aux_d[place+N*N],-G_lin_d[place]*(x_vec_lin_d[1+i*2+steps*2*place]*x_vec_lin_d[1+i*2+steps*2*place]));
			/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		}
	}
}


void fillA_c(arma::Mat<double> &A,int N,boost::mt19937 &rng,int caso)
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

void fillG_c(std::vector<double> &G,int N,boost::mt19937 &rng,int caso)
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


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void calculateStuff_c(arma::Mat<double> A,int steps, double *x_vec_lin,int N,std::vector<double> G,double *flux_aux_d) //1 tiempo. 2 posicion. 3 momento. 4 energia potencial. 5 energia cinetica. 6 energia. 7 energia Total
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

double calcdt_c(int N)
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

void itera_c(arma::Mat<double> &A,std::vector<double> &G,int N,double T_t, double StartPoint,arma::Mat<double> &T,int place)
{
	double FinishPoint=1-StartPoint;
	double T_t_i=T_t;
	T_t=T_t*FinishPoint;
	double dt=calcdt_c(N);
	printf("%lf\n",dt );

	int pasos=1+(int)((double)(2.0*N+N*N+N*(N+1.0)+2.0*N*T_t/dt)/70000000.0);
	printf("%d\n",pasos );

	int steps=T_t/(dt*pasos);
	printf("%d\n", steps );

//////////////////////////////////////////////////////////////////////////////////////////////////
	double *flux_aux;
	//double *flux_aux_aux;
	double *flux_aux_d;
	double *x_vec_lin;
	///
	flux_aux=(double*)malloc(sizeof(double)*N*(N+1));
	//flux_aux_aux=(double*)malloc(sizeof(double)*N*(N+1));
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
		calculateStuff_c(A,steps,x_vec_lin,N,G,flux_aux_d);
		/*if(cudaMemcpy(flux_aux_aux, flux_aux_d, (N+1)*N*sizeof(double), cudaMemcpyDeviceToHost)!=cudaSuccess)
		{
			printf("fluxmal\n");
			return;
		}
		for (int i = 0; i < N; ++i)
		{
			for (int j = 0; j < N+1; ++j)
			{
				flux_aux[i+N*j]=flux_aux[i+N*j]+flux_aux_aux[i+N*j];
			}
		}*/
		printf("cudacall (%d/%d) ended\n",k+1,pasos);
    }
    fclose(f);

	if(cudaMemcpy(flux_aux, flux_aux_d, (N+1)*N*sizeof(double), cudaMemcpyDeviceToHost)!=cudaSuccess)
	{
		printf("fluxmal\n");
		return;
	}
	printf("copied to memory\n");
	printf("%.15lf\n",flux_aux[0+N*1]);
	printf("%.15lf\n",flux_aux[1+N*0]);

    char savename1[80]={};
    sprintf(savename1,"T_%d.txt",place);
	FILE *g=fopen(savename1,"w");
	for( int i=0; i<N; ++i )
	{

		for (int j = 0; j < N+1; ++j)
		{
			T(i,j)=flux_aux[i+N*j];
			fprintf(g,"%.10lf	  ",flux_aux[i+N*j]); //1 posicion. 2 momento. 3 energia potencial. 4 energia cinetica. 5 energia total
		}
		if(flux_aux[i+N*N]>0)
		{
			printf("%d tiene su gamma alreves\n", i);
		}
		fprintf(g, "\n");
	}
	printf("factor %lf!\n",T_t/dt);
}

void getT_c(int N, double T_t, double StartPoint,boost::mt19937 &rng,arma::Mat<double> &T,int place)
{
    using namespace std;
    arma::Mat<double> A(N,N);
    vector<double> G(N);
//////////////////////////////////////////////////////////////////////////
	fillA_c(A,N,rng,1);
	fillG_c(G,N,rng,1);
////////////////////////////////////////////////////////////////////////////
	itera_c(A,G,N,T_t,StartPoint,T,place);
	printf("N (cuda)=%d\n",N);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


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

void inicialcond_b(state_type &x,int N,boost::mt19937 &rng,int caso,std::vector<state_type> x_vec,size_t steps)
{
    boost::uniform_real<> unif( 0, 0.01);//la distribucion de probabilidad uniforme entre cero y 2pi
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
    if(caso==3)
    {
    	FILE *w= fopen("Xi.txt", "w");
    	for (int i = 0; i < N; ++i)
		{
			fprintf(w, "%f  ",x_vec[steps][2*i] );
			fprintf(w, "%f\n",x_vec[steps][2*i+1] );
		}
		fclose(w);
    }
}

void fillA_b(arma::Mat<double> &A,int N,boost::mt19937 &rng,int caso)
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

void fillG_b(std::vector<double> &G,int N,boost::mt19937 &rng,int caso)
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


void fillI_b(std::vector<double> &I,int N,boost::mt19937 &rng,int caso)
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

void fillFw_b(std::vector<double> &Fw,int N,boost::mt19937 &rng,int caso)
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

void fillW_b(std::vector<double> &Fw,int N,boost::mt19937 &rng,int caso)
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

void printsave_b(size_t steps, std::vector< state_type > &x_vec,std::vector<double> &times,int N) //1 tiempo. 2 posicion. 3 momento. 4 energia potencial. 5 energia cinetica. 6 energia. 7 energia Total
{

	FILE *f=fopen("save.txt","a");

	for( size_t i=0; i<steps; ++i )
	{
		if(i%(steps/100)==0 && i < steps)
		{
			printf("printing savestate: %d \n", (int)(100.0*i/steps));
		}
		fprintf(f,"%lf  ",times[i] );
		for (int j = 0; j < N; ++j)
		{
			fprintf(f,"%.10lf	  %.10lf   ",x_vec[i][2*j],x_vec[i][2*j+1]); //1 posicion. 2 momento. 3 energia potencial. 4 energia cinetica. 5 energia total
		}
		fprintf(f,"\n");
	}	
	fclose(f);
}

void itera_b(double t_in, double t_fn,double dt,arma::Mat<double> &A,std::vector<double> &I,std::vector<double> &G,std::vector<double> &F,std::vector<double> &Fw,int N,int load,boost::mt19937 &rng)
{
    using namespace std;
    using namespace boost::numeric::odeint;
    ///////////////////////////////////////////////////////////////////
	double K=1;
	size_t steps;
	state_type x(2*N); //condiciones iniciales
    vector<state_type> x_vec;
    vector<double> times;
    if(t_in<1)
    {
		inicialcond_b(x,N,rng,0,x_vec,steps);
    }
    else
    {
		inicialcond_b(x,N,rng,1,x_vec,steps);
    }

///////////////////////////////////////////////////////////////////////
	harm_osc ho(K,N,I,A,F,G,Fw);
    runge_kutta4<
                      state_type , double ,
                      state_type , double ,
                      openmp_range_algebra
                    > stepper;
    int chunk_size = N/omp_get_max_threads();
    omp_set_schedule( omp_sched_static , chunk_size );
    printf("solving..\n");
	steps = integrate_adaptive(stepper, ho, x , t_in , t_fn , dt,push_back_state_and_time( x_vec , times )); //1 funcion. 2 condiciones iniciales. 3 tiempo inicial. 4 tiempo final. 5 dt inicial. 6 vector de posicion y tiempo
	printsave_b(steps,x_vec,times,N);
	inicialcond_b(x,N,rng,3,x_vec,steps);
}


void solve_b(int N,double T_t,int load,boost::mt19937 &rng,double dt)
{
    using namespace std;
    arma::Mat<double> A(N,N);
    vector<double> I(N);
    vector<double> G(N);
    vector<double> F(N);
    vector<double> Fw(N);
	fillA_b(A,N,rng,1);
	fillG_b(G,N,rng,load);
	fillI_b(I,N,rng,load);
	fillFw_b(F,N,rng,load);
	fillW_b(Fw,N,rng,load);
	FILE *c=fopen("save.txt","w");
	fclose(c);
////////////////////////////////////////////////////////////////////////////
	int number_of_partitions=1+(int)((double)(T_t/dt)*2*N/10000000);
	printf("%d\n",number_of_partitions );
	double t_in=0.0;
	double t_fn;
	for (int i = 0; i < number_of_partitions; ++i)
	{
		printf("part (%d/%d)\n",i+1,number_of_partitions );
		t_fn=t_in+(T_t/number_of_partitions);
		itera_b(t_in,t_fn,dt,A,I,G,F,Fw,N,load,rng);
		t_in=t_fn;
	}
	printf("N (boost)=%d\n",N);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void generateKdist(int N,boost::mt19937 &rng, std::vector<int> &K_dist)
{
    boost::uniform_int<> unif( 3, 3);//la distribucion de probabilidad uniforme entre cero y 2pi
    boost::variate_generator< boost::mt19937&, boost::uniform_int<> > gen( rng , unif );//gen es una funcion que toma el engine y la distribucion y devuelve el numero random
    boost::uniform_real<> unif2( 0, 1 );//la distribucion de probabilidad uniforme entre cero y 2pi
    boost::variate_generator< boost::mt19937&, boost::uniform_real<> > gen2( rng , unif2 );//gen es una funcion que toma el engine y la distribucion y devuelve el numero random
    FILE *f;
    int stubs_sum=1;

    int caso=0;
    if(caso==0)
    {
    	while(stubs_sum%2==1)
    	{
    		stubs_sum=0;
    		f=fopen("K_dist","w");
    		for (int i = 0; i < N; ++i)
    		{	
    			K_dist[i]=gen();
    			stubs_sum=stubs_sum+K_dist[i];
				fprintf(f, "%d  ", K_dist[i]);
    		}
    		fclose(f);
    		//printf("%d\n",stubs_sum );
    	}
    }
}

void checkitera_a(arma::Mat<int> &K,int N,int i,int j)
{
	K(i,j)=2;
	K(j,i)=2;
	for (int k = 0; k < N; ++k)
	{
		if(K(j,k)==1)
		{
			//printf("asd %d  %d\n",j,k);
			checkitera_a(K,N,j,k);

		}
	}
}

int check_a(arma::Mat<int> &K,int N)
{
    arma::Mat<int> B(N,N);
   	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			B(i,j)=K(j,i);
		}
	}
	checkitera_a(B,N,0,0);
   	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			if(B(i,j)==1)
			{
				return 0;
			}
		}
	}
	return 1;
}

int fillK_a(arma::Mat<int> &K,int N,boost::mt19937 &rng,int caso)
{
    boost::uniform_int<> unif( 1, N);//la distribucion de probabilidad uniforme entre cero y 2pi
    boost::variate_generator< boost::mt19937&, boost::uniform_int<> > gen( rng , unif );//gen es una funcion que toma el engine y la distribucion y devuelve el numero random
    boost::uniform_real<> unif2( 0, 1 );//la distribucion de probabilidad uniforme entre cero y 2pi
    boost::variate_generator< boost::mt19937&, boost::uniform_real<> > gen2( rng , unif2 );//gen es una funcion que toma el engine y la distribucion y devuelve el numero random
    FILE *f;
    std::vector<int> K_dist(N);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine e(seed);

    int stubs_sum=1;

    if(caso==0)
    {
    	f=fopen("K_dist","r");
		for (int i = 0; i < N; ++i)
		{
			fscanf(f, "%d", &K_dist[i]);
		}
		fclose(f);
    }
    if(caso==1)
    {
    	generateKdist(N,rng,K_dist);
    }

    for (int i = 0; i < N; ++i)
    {
    	for (int j = 0; j < N; ++j)
    	{
    		K(i,j)=0;
    		if(i==j)
    		{
    			K(i,j)=1;
    		}
    	}
    }
	std::shuffle(std::begin(K_dist), std::end(K_dist), e);
    int current;
    int suitables;
    std::vector<int> suitables_list(N);
    int maxstubs=0;
    int maxstubs_place;
    for (int l = 0; l < N; ++l)
	{
		maxstubs=0;
		for (int i = 0; i < N; ++i)
		{
			if(K_dist[i]>maxstubs)
			{
				maxstubs=K_dist[i];
				maxstubs_place=i;
			}
		}
		if(maxstubs==0)
		{
			return 1;
		}
		int i=maxstubs_place;
		//printf("%d\n", i);
		std::fill(suitables_list.begin(), suitables_list.end(), 0);
		while(K_dist[i]>0)
		{
			suitables=0;
			for (int j = 0; j < N; ++j)
			{
				if(j!=i && K_dist[j]>0 && K(i,j)==0)
				{
					suitables_list[suitables]=j;
					suitables=suitables+1;
				}
			}
			if(suitables==0)
			{
				return 0;
			}
			current=suitables_list[(int)(suitables*gen2())];
			//printf("%d   %d   %d   %d    %d\n",i,K_dist[i],current,K_dist[current], K(i,current) );
			if(K_dist[i]>0 && K_dist[current]>0 && K(i,current)==0 && i!=current)
			{
				K(i,current)=1;
				K(current,i)=1;
				K_dist[i]=K_dist[i]-1;
				K_dist[current]=K_dist[current]-1;
			}

		}
	}
	return 1;
/////////////
///////////
}

void getA_a(boost::mt19937 &rng,int N,int caso2,int place)
{
    const rlim_t kStackSize = 100 * 1024 * 1024;   // min stack size = 16 MB
    struct rlimit rl;
    int result;

    result = getrlimit(RLIMIT_STACK, &rl);
    if (result == 0)
    {
        if (rl.rlim_cur < kStackSize)
        {
            rl.rlim_cur = kStackSize;
            result = setrlimit(RLIMIT_STACK, &rl);
            if (result != 0)
            {
                fprintf(stderr, "setrlimit returned result = %d\n", result);
            }
        }
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////
    char savename1[80]={};
    char savename2[80]={};

    //////////////////////////////////////////////////////////////////////////////////
    using namespace std;

///////////////////////////////////////////////////////////////////////
    FILE *f1;
    FILE *f2;

    arma::Mat<int> K(N,N);
////////////////////////////////////////////////////////////////////////////
        sprintf(savename1,"Ai.txt");
    	f1=fopen(savename1,"w");
        sprintf(savename2,"Neighbors_%d.txt",place);
    	f2=fopen(savename2,"w");
		while(1==1)
		{
			if(fillK_a(K,N,rng,caso2)==0)
			{
				printf("fail\n");
				continue;
			}
			//printf("succes\n");
			if(check_a(K,N)==1)
			{
				for (int i = 0; i < N; ++i)
				{
					for (int j = 0; j <= i; ++j)
					{
						if(i==j)
						{
							fprintf(f1, "%d  ",2*K(i,j));
						}
						else
						{
							fprintf(f1, "%d  ",K(i,j));
						}
						
					}
				}
				for (int i = 0; i < N; ++i)
				{
					for (int j = 0; j < N; ++j)
					{
						if(K(i,j)==1 && i!=j)
						{
							fprintf(f2, "%d  ",j);
						}

					}
					fprintf(f2,"\n");
				}
				break;
			}
		}
    	fclose(f1);
    	fclose(f2);
    

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double mediann_ca(int N,arma::Mat<double> &A,int j,std::vector<int> &capes)
{
	double medi=0;
	double meditotal=0;
	int cap=0;
	for (int i = 0; i < N; ++i)
	{
		medi=0;
		if(capes[i]==j)
		{
			//printf("%d    %d\n",i,j );
			cap=cap+1;
			for (int k = 0; k < N; ++k)
			{
				if(A(i,k)>0 && i!=k)
				{
					medi=medi+1;
				}
			}

		}
		meditotal=meditotal+medi;
	}
	meditotal=meditotal/cap;
	return(meditotal);
}

void capas_ca(int N,arma::Mat<double> &A,std::vector<double> &G,std::vector<int> &Caps,int place)
{
	std::vector<int> capes(N,-1);
	capes[0]=0;
	int Ntotal=1;
	int Ncapas=0;
	char savename1[80]={};
    sprintf(savename1,"capas_%d.txt",place);


		for (int j = 0; j < N; ++j) //capas
		{
			if(Ntotal==N)
			{
				break;
			}
			for (int i = 0; i < N; ++i) //
			{
				if(capes[i]==j)
				{
					for (int k = 0; k < N; ++k)
					{
						if(A(i,k)>0 && capes[k]<0)
						{

							capes[k]=j+1;
							Ntotal=Ntotal+1;
						}
					}
				}

			}
			
		}
	
	printf("%d=%d\n", Ntotal,N );

	for (int i = 0; i < N; ++i)
	{
		if(Ncapas<capes[i])
		{
			Ncapas=capes[i];
		}
	}
	FILE *w= fopen(savename1, "w");
	for (int i = 0; i <= Ncapas; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			if(capes[j]==i)
			{
				fprintf(w, "%d   ",j);
			}
		}
		fprintf(w, "\n" );
	}
	fclose(w);
								
	double median=0;
	int Cnumber=0;
	for (int i = 0; i <= Ncapas; ++i)
	{
		median=0;
		Cnumber=0;

		for (int j = 0; j < N; ++j)
		{
			if(capes[j]==i)
			{
				median=median+G[j];
				Cnumber=Cnumber+1;
			}
		}
		printf("Capa %d  <G>=%lf\n",i,median/Cnumber );
		printf("Capa %d  N=%d\n",i, Cnumber );
		printf("Capa %d  <N>=%lf\n",i, mediann_ca(N,A,i,capes) );
	}
	for (int i = 0; i < N; ++i)
	{
		Caps[i]=capes[i];
	}
}


void fillG_ca(std::vector<double> &G,int N)
{

		FILE *r= fopen("Gi.txt", "r");
		for (int i = 0; i < N; ++i)
		{
			fscanf(r, "%lf", &G[i]);
		}
		fclose(r);
}


void changeA_ca(int N,double K, double kappa,std::vector<int> &Caps,int place)
{
    int test=0;

    arma::Mat<double> A(N,N);
    std::vector<double> G(N);
	fillG_ca(G,N);

    int thinness=0;

////////////////////////////////////////////////////////////////////////////

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
    if(test!=1)
    {
	FILE *t= fopen("Thinness.txt", "w");
    for (int i = 0; i < N; ++i)
	{
		thinness=0;
		for (int j = 0; j < N; ++j)
		{
			if(i==j)
			{
				A(i,j)=kappa;
			}
			if(i != j && A(i,j)>0)
			{
				A(i,j)=K;
				thinness=thinness+1;
			}
		}
		fprintf(t, "%d \n ",thinness);
	}
	fclose(t);

    FILE *w= fopen("Ai.txt", "w");
    for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j <= i; ++j)
		{
			fprintf(w, "%f  ",A(i,j));
		}
	}
	fclose(w);
    }

	capas_ca(N,A,G,Caps,place);

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Tproperties(arma::Mat<double> &P,int N,arma::Mat<double> &T,std::vector<int> &Caps)
{
	int max=0;
	for( int i=0; i<N; ++i )
	{
		if(max<=Caps[i])
		{
			max=Caps[i];
		}

	}
	FILE *f=fopen("Gi.txt","r");
    std::vector<double> G(N);
	for( int i=0; i<N; ++i )
	{
		fscanf(f,"%lf ",&G[i]); //1 posicion. 2 momento. 3 energia potencial. 4 energia cinetica. 5 energia total
	}
	fclose(f);
	int N_i;
	int N_i_in;
	int N_i_out;
	double j_i;
	double j_i_in;
	double j_i_out;
	double j_i_prev;
	double j_i_next;
	for (int i = 0; i < N; ++i)
	{
		N_i=0;
		N_i_in=0;
		N_i_out=0;
		j_i=0;
		j_i_in=0;
		j_i_out=0;
		j_i_prev=0;
		j_i_next=0;
		for (int j = 0; j < N; ++j)
		{
			if(i!=0)
			{
				if(Caps[j]==Caps[i-1])
				{
					j_i_prev=j_i_prev+T(i,j);
				}
			}
			if(i!=N-1)
			{
				if(Caps[j]==Caps[i+1])
				{
					j_i_next=j_i_next+T(i,j);
				}
			}
			if(T(i,j)>0.00000000000001)
			{
				N_i=N_i+1;
				N_i_in=N_i_in+1;
				j_i=j_i+T(i,j);
				j_i_in=j_i_in+T(i,j);
			}
			if(T(i,j)<-0.00000000000001)
			{
				N_i=N_i+1;
				N_i_out=N_i_out+1;
				j_i=j_i+T(i,j);
				j_i_out=j_i_out+T(i,j);
			}
		}
		P(i+N*Caps[i],0)=G[i];
		P(i+N*Caps[i],1)=P(i+N*Caps[i],1)+N_i;
		P(i+N*Caps[i],2)=P(i+N*Caps[i],2)+N_i_in;
		P(i+N*Caps[i],3)=P(i+N*Caps[i],3)+N_i_out;
		P(i+N*Caps[i],4)=P(i+N*Caps[i],4)+(double)N_i_in/N_i;
		P(i+N*Caps[i],5)=P(i+N*Caps[i],5)+j_i;
		P(i+N*Caps[i],6)=P(i+N*Caps[i],6)+j_i_in;
		P(i+N*Caps[i],7)=P(i+N*Caps[i],7)+j_i_out;
		P(i+N*Caps[i],8)=P(i+N*Caps[i],8)+j_i_in/(j_i_in-j_i_out);
		P(i+N*Caps[i],9)=P(i+N*Caps[i],9)+T(i,N);
		P(i+N*Caps[i],10)=P(i+N*Caps[i],12)+j_i_prev;
		P(i+N*Caps[i],11)=P(i+N*Caps[i],13)+j_i_next;
		P(i+N*Caps[i],12)=P(i+N*Caps[i],10)+1;
		P(i+N*Caps[i],13)=Caps[i];
	}
}

void saveP(arma::Mat<double> &P,int N, int n_total,int place, FILE *gplotpipe)
{
    FILE *f=fopen("P.txt","w");
    for (int i = 0; i < N*N; ++i)
	{
		if(P(i,n_total-2)>0.1)
		{
			for (int j = 0; j < n_total; ++j)
			{
				if(j>0 && j<n_total-2 )
				{
					fprintf(f, "%.10lf  ",(double)P(i,j)/P(i,n_total-2));
				}
				
				else
				{
					fprintf(f, "%.10lf  ",P(i,j));
				}
			}
			fprintf(f, "\n");
		}

	}
	fclose(f);
    FILE *g=fopen("P_save.txt","w");
    for (int i = 0; i < N*N; ++i)
	{
			for (int j = 0; j < n_total; ++j)
			{
				fprintf(g, "%.10lf  ",P(i,j));
			}
			fprintf(g, "\n");
	}
	fclose(g);
	FILE *fp=fopen("save_place.txt","w");
	fprintf(fp, "%d\n",place);
	fclose(fp);
	fprintf(gplotpipe, "splot 'P.txt' u 1:(log($12)):14 w p palette pt 7 ps 0.5 \n" );
	pclose(gplotpipe);
}

int loadP(arma::Mat<double> &P,int N, int n_total)
{
    FILE *f=fopen("P_save.txt","r");
    for (int i = 0; i < N*N; ++i)
	{
			for (int j = 0; j < n_total; ++j)
			{
				fscanf(f, "%lf",&P(i,j));
			}
	}
    fclose(f);
    FILE *fp=fopen("save_place.txt","r");
    int place;
    fscanf(fp, "%d",&place);
    fclose(fp);
    return(place);
}

int main()
{
    using namespace std;
    boost::mt19937 rng(static_cast<unsigned int>(std::time(0)));  /// el engine para generar numeros random

    FILE *gplotpipe;

//////////////////////////////////////////////////////////
    int N;
    printf("N: ");
    std::cin >>N;

    double T_t;
    printf("Total Time: ");
    std::cin >>T_t;

    double dt_b;
    printf("dt (0.01): ");
    std::cin >>dt_b;

    double K;
    printf("K: ");
    std::cin >>K;

    double kappa;
    printf("Kappa: ");
    std::cin >>kappa;

    double StartPoint_c;
    printf("Start Point (cuda) [x:1]: ");
    std::cin >>StartPoint_c;

    int caso;
    printf("Generate Kdist? (0 NO:: 1 YES):  ");
    std::cin >>caso;

    int casoP;



    int loop;
    if(caso==0)
    {
    	printf("Load P? (0 NO:: 1 YES):  ");
    	std::cin >>casoP;
    	printf("Loops: ");
    	std::cin >>loop;
    }
    else
    {
    	casoP=0;
    	loop=1;
    }

    int n_total=14;
    arma::Mat<double> P(N*N,n_total);
    arma::Mat<double> T(N,N+1);
    std::vector<int> Caps(N);
    P.zeros();
    int place=0;
    if(casoP==1)
    {
    	place=loadP(P,N,n_total);
    	if(place>=loop)
    	{
    		printf("place: %d >= %d\n",place,loop );
    	}
    }

///////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    for (int i = place; i < loop; ++i)
    {
		gplotpipe= popen("gnuplot -p", "w");
    	fprintf(gplotpipe, "set terminal png size 1022,606\n");
    	fprintf(gplotpipe, "set yr[-15:5]\n" );
    	fprintf(gplotpipe, "set view map\n" );
        fprintf(gplotpipe, "set output 'Jnext_%d.png'\n",i );
    	printf("loop (%d/%d)\n",i+1,loop );
    	getA_a(rng,N,caso,i);
		changeA_ca(N,K,kappa,Caps,i);

		int caso2;
		if(caso==1)
		{
			caso2=0;
		}
		else
		{
			caso2=1;
		}
    	solve_b(N,T_t,caso2,rng,dt_b); //el 0 no carga condiciones iniciales
    	getT_c(N,T_t,StartPoint_c,rng,T,i);
    	Tproperties(P,N,T,Caps);
    	printf("loop #%d\n",loop );
    	saveP(P,N,n_total,i+1,gplotpipe);
    }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	return 0;
}