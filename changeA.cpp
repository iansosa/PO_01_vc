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

int main()
{
    int N;
    printf("N: ");
    std::cin >>N;
    arma::Mat<double> A(N,N);

    double kappa;
    printf("Kappa: ");
    std::cin >>kappa;

    double K;
    printf("K: ");
    std::cin >>K;

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

////////////////////////////////////////////////////////////////////////////

	/*printstuff(K,steps,x_vec,times,M,N,L,B,G);
	system("cat ac_1.txt ac_2.txt ac_3.txt ac_4.txt> ac.txt");*/



	return 0;
}