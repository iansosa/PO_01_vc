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
#include <sys/resource.h>
#include <omp.h>
#include <boost/numeric/odeint.hpp>
#include <boost/random.hpp>

int checkitera_T_forward(std::vector<int> &H,int N,int j,arma::Mat<double> &T)
{
	if(H[j]==1)
	{
		return 1;
	}
	H[j]=1;
	for (int k = 0; k < N; ++k)
	{
		if(T(j,k)<-0.00000000000001)
		{
			//printf("asd %d  %d\n",j,k);
			return checkitera_T_forward(H,N,k,T);

		}
	}
	return 0;
}

int checkitera_T_backwards(std::vector<int> &H,int N,int j,arma::Mat<double> &T)
{
	if(H[j]==1)
	{
		return 1;
	}
	H[j]=1;
	for (int k = 0; k < N; ++k)
	{
		if(T(j,k)>0.00000000000001)
		{
			//printf("asd %d  %d\n",j,k);
			return checkitera_T_backwards(H,N,k,T);

		}
	}
	return 0;
}

int check_T(arma::Mat<double> &T,int N,int start,int caso)
{
	std::vector<int> H(N);
	std::fill(H.begin(), H.end(), 0);

	if(caso==-1)
	{
		return checkitera_T_backwards(H,N,start,T);
	}
	if(caso==1)
	{
		return checkitera_T_forward(H,N,start,T);
	}
	printf("wrong caso\n");
	return 0;
	
}

int checkforcicle(arma::Mat<double> &T,int N,int promedio)
{
	int caso=0;
	for (int i = 0; i < N; ++i)
	{
		if(check_T(T,N,i,-1)==1)
		{
			printf("%d tiene ciclo (backwards) en T_%d!!\n",i,promedio);
			caso=-1;
		}
		if(check_T(T,N,i,1)==1)
		{
			printf("%d tiene ciclo (forward) en T_%d!!\n",i,promedio);
			caso=1;
		}
	}
	return caso;
}

void loadT(int N,arma::Mat<double> &T,FILE *f)
{
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N+1; ++j)
		{
			fscanf(f, "%lf", &T(i,j));
			//printf("%lf\n",T(i,j) );
		}
	}
}

void Stats(int N,int k_max,int N_stats,arma::Mat<double> &T,std::vector<double> &S)
{

    std::vector<double> S_aux(N_stats);
	std::fill(S_aux.begin(), S_aux.end(), 0);
	int n_intake;
	for (int i = 0; i < N; ++i)
	{
		n_intake=0;
		for (int j = 0; j < N; ++j)
		{

			if(T(i,j)<-0.00000000000001 && i!=j)
			{
				n_intake=n_intake+1;
			}
		}
		S_aux[n_intake]=S_aux[n_intake]+1;
	}

	for (int i = 0; i <= k_max; ++i)
	{
		S_aux[i]=S_aux[i]/N;
		S[i]=S[i]+S_aux[i];
	}
}

void printStats(int N_stats,std::vector<double> &S)
{
	for (int i = 0; i < N_stats; ++i)
	{
		printf("S[%d]=%lf   \n",i,S[i]);
	}
}

int main()
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

    int N;
    printf("N: ");
    std::cin >>N;

    int total;
    printf("T_[x] (maximo): ");
    std::cin >>total;

    int k_max;
    printf("k (maximo): ");
    std::cin >>k_max;

    int N_stats;
    printf("Stats (N_in(0),..,N_in(k),x):  (k+1+x) ");
    std::cin >>N_stats;

    if(N_stats<k_max+1)
    {
    	printf("(Error) N_stats<k_max+1\n");
    	return 0;
    }

    char savename[80]={};
    FILE *f;

    arma::Mat<double> T(N,N+1);

    std::vector<double> S(N_stats);
	std::fill(S.begin(), S.end(), 0);
	int cicle=0;

    for (int i = 0; i <= total; ++i)
    {
		sprintf(savename,"T_%d.txt",i);
		f=fopen(savename,"r");
		loadT(N,T,f);
		Stats(N,k_max,N_stats,T,S);
		if(checkforcicle(T,N,i)!=0)
		{
			//printf("hay ciclo en T_%d!!\n",i);
			cicle=1;
		}
		fclose(f);
    }
    if(cicle==0)
    {
    	printf("no hay ciclo\n");
    }
    for (int i = 0; i < N_stats; ++i)
    {
    	S[i]=S[i]/(total+1);
    }
    printStats(N_stats,S);
	return 0;
}