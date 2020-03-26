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
#include <boost/random.hpp>

typedef std::vector< double > state_type;

void fillK(arma::Mat<double> &K,int N,boost::mt19937 &rng,double set)
{
    boost::uniform_real<> unif( 0, 1 );//la distribucion de probabilidad uniforme entre cero y 2pi
    boost::variate_generator< boost::mt19937&, boost::uniform_real<> > gen( rng , unif );//gen es una funcion que toma el engine y la distribucion y devuelve el numero random

    for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j <= i; ++j)
		{
			if(gen()>=set || j==i)
			{
				K(i,j)=1;
			}
			else
			{
				K(i,j)=0;
			}

		}
	}
   	for (int i = 0; i < N; ++i)
	{
		for (int j = N-1; j > i; --j)
		{
			K(i,j)=K(j,i);
		}
	}
	/////////////
///////////
}

void checkitera(arma::Mat<double> &K,int N,int i,int j)
{
	K(i,j)=2;
	for (int k = 0; k < N; ++k)
	{
		if(K(j,k)==1)
		{
			checkitera(K,N,j,k);
		}
	}
}

int check(arma::Mat<double> &K,int N)
{
    arma::Mat<double> B(N,N);
   	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			B(i,j)=K(j,i);
		}
	}
	checkitera(B,N,0,0);
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
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main()
{
    char savename1[80]={};
    char savename2[80]={};

    //////////////////////////////////////////////////////////////////////////////////
    using namespace std;
    boost::mt19937 rng(static_cast<unsigned int>(std::time(0))); 

///////////////////////////////////////////////////////////////////////
    FILE *f1;
    FILE *f2;
    int N;
    printf("N: ");
    std::cin >>N;
    double set;
    arma::Mat<double> K(N,N);
    int npasos;
    printf("npasos: ");
    std::cin >>npasos;
////////////////////////////////////////////////////////////////////////////
    for (int j = 0; j < npasos; ++j)
    {
    	printf("porcentaje: %lf\n", (double)100.0*j/npasos );
    	set=(double)(j)/npasos;
        sprintf(savename1,"Ai_%.3lf.txt",set);
    	f1=fopen(savename1,"w");
        sprintf(savename2,"Neighbors_%.3lf.txt",set);
    	f2=fopen(savename2,"w");
		while(1==1)
		{
			fillK(K,N,rng,set);
			if(check(K,N)==1)
			{
				for (int i = 0; i < N; ++i)
				{
					for (int j = 0; j <= i; ++j)
					{
						if(i==j)
						{
							fprintf(f1, "%lf  ",2*K(i,j));
						}
						else
						{
							fprintf(f1, "%lf  ",K(i,j));
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


////////////////////////////////////////////////////////////////////////////

	/*printstuff(K,steps,x_vec,times,M,N,L,B,G);
	system("cat ac_1.txt ac_2.txt ac_3.txt ac_4.txt> ac.txt");*/



	return 0;
}