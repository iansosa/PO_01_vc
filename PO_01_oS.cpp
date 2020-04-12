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



void inicialcond(state_type &x,int N,boost::mt19937 &rng,int caso,std::vector<state_type> x_vec,size_t steps)
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

void printsave(size_t steps, std::vector< state_type > &x_vec,std::vector<double> &times,int N) //1 tiempo. 2 posicion. 3 momento. 4 energia potencial. 5 energia cinetica. 6 energia. 7 energia Total
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

void itera(double t_in, double t_fn,double dt,arma::Mat<double> &A,std::vector<double> &I,std::vector<double> &G,std::vector<double> &F,std::vector<double> &Fw,int N,int load)
{
    using namespace std;
    using namespace boost::numeric::odeint;
    boost::mt19937 rng(static_cast<unsigned int>(std::time(0)));
    ///////////////////////////////////////////////////////////////////
	double K=1;
	size_t steps;
	state_type x(2*N); //condiciones iniciales
    vector<state_type> x_vec;
    vector<double> times;
    if(t_in<1 && load !=1)
    {
		inicialcond(x,N,rng,0,x_vec,steps);
    }
    else
    {
		inicialcond(x,N,rng,1,x_vec,steps);
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
	printsave(steps,x_vec,times,N);
	inicialcond(x,N,rng,3,x_vec,steps);
}

int main()
{
    using namespace std;
    boost::mt19937 rng(static_cast<unsigned int>(std::time(0)));
////////////////////////////////////////////////////////////////////////
    int N;
    printf("N: ");
    std::cin >>N;

    int load;
    printf("Load CI (0 NO, 1 YES): ");
    std::cin >>load;

    double dt;
    printf("dt (0.01): ");
    std::cin >>dt;

    double T_t;
    printf("Total time: ");
    std::cin >>T_t;


    arma::Mat<double> A(N,N);
    vector<double> I(N);
    vector<double> G(N);
    vector<double> F(N);
    vector<double> Fw(N);

//////////////////////////////////////////////////////////////////////////
	fillA(A,N,rng,load);
	fillG(G,N,rng,load);
	fillI(I,N,rng,load);
	fillFw(F,N,rng,load);
	fillW(Fw,N,rng,load);
	FILE *c=fopen("save.txt","w");
	fclose(c);
////////////////////////////////////////////////////////////////////////////
	int number_of_partitions=1+(int)((double)(T_t/dt)*2*N/20000000);
	printf("%d\n",number_of_partitions );
	double t_in=0.0;
	double t_fn;
	for (int i = 0; i < number_of_partitions; ++i)
	{
		printf("part (%d/%d)\n",i+1,number_of_partitions );
		t_fn=t_in+(T_t/number_of_partitions);
		itera(t_in,t_fn,dt,A,I,G,F,Fw,N,load);
		t_in=t_fn;
	}
	printf("N=%d\n",N);

	return 0;
}