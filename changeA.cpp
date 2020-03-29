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

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void fillG(std::vector<double> &G,int N)
{

		FILE *r= fopen("Gi.txt", "r");
		for (int i = 0; i < N; ++i)
		{
			fscanf(r, "%lf", &G[i]);
		}
		fclose(r);
}

void capas(int N,arma::Mat<double> &A,std::vector<double> &G)
{
	std::vector<int> capes(N,-1);
	capes[0]=0;
	int Ntotal=1;
	int Ncapas=0;

	FILE *w= fopen("capas.txt", "w");
	fprintf(w, "%d\n", 0);
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
							fprintf(w, "%d   ",k);
							capes[k]=j+1;
							Ntotal=Ntotal+1;
						}
					}
				}

			}
			fprintf(w, "\n" );
		}
	printf("%d=%d\n", Ntotal,N );

	for (int i = 0; i < N; ++i)
	{
		if(Ncapas<capes[i])
		{
			Ncapas=capes[i];
		}
	}

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
	}
}


int main()
{
    int N;
    printf("N: ");
    std::cin >>N;
    int test;
    printf("Test? (1 YES): ");
    std::cin >>test;

    arma::Mat<double> A(N,N);
    std::vector<double> G(N);
	fillG(G,N);

    double kappa;
    double K;
	if(test!=1)
	{

    printf("Kappa: ");
    std::cin >>kappa;


    printf("K: ");
    std::cin >>K;
	}


    int thinness=0;

////////////////////////////////////////////////////////////////////////////
    if(test!=1)
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

	capas(N,A,G);

////////////////////////////////////////////////////////////////////////////

	/*printstuff(K,steps,x_vec,times,M,N,L,B,G);
	system("cat ac_1.txt ac_2.txt ac_3.txt ac_4.txt> ac.txt");*/



	return 0;
}