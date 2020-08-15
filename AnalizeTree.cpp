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

int mini_checktree(arma::Mat<double> &A,int N,int prev,int id,int &ntotal,int longest)
{
	ntotal=ntotal+1;
	longest=longest+1;
	int aux=longest;
	int size;
	for (int i = 1; i < N; ++i)
	{
		//printf("A(%d,%d)=%lf\n",id,i,A(id,i) );
		if(A(id,i)>0.0000001 && i!=id && i!=prev)
		{
			//printf("asdafs\n");
			size=mini_checktree(A,N,id,i,ntotal,longest);
			if(aux<size)
			{
				//printf("asd\n");
				aux=size;
			}
		}
	}
	return aux;
	
}

void checktree(arma::Mat<double> &A,int N,int id,int &ntotal, int &longest)
{
	ntotal=1;
	longest=1;
	int size;
	for (int i = 1; i < N; ++i)
	{
		//printf("A(%d,%d)=%lf checktree\n",id,i,A(id,i) );
		if(A(i,id)>0.0000001 && i!=id)
		{
			//printf("hola\n");
			size=mini_checktree(A,N,id,i,ntotal,1);
			if(longest<size)
			{
				longest=size;
			}
		}
	}
}

void loadA(int N,arma::Mat<double> &A,FILE *f)
{
    	for (int i = 0; i < N; ++i)
		{
			for (int j = 0; j <= i; ++j)
			{
				fscanf(f,"%lf",&A(i,j));
			}
		}
    	for (int i = 0; i < N; ++i)
		{
			for (int j = N-1; j > i; --j)
			{
				A(i,j)=A(j,i);
			}
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

	arma::Mat<double> A(N,N);

    char savename[80]={};
    FILE *f;
	sprintf(savename,"Ai.txt");
	f=fopen(savename,"r");
	loadA(N,A,f);

while(1==1)
{

    int oscid;
    printf("Oscilator ID: ");
    std::cin >>oscid;

    int ntotal;
    int longest;
    checktree(A,N,oscid,ntotal,longest);

    printf("Ntotal=%d  Longest=%d\n", ntotal,longest);



    int done=0;
    printf("Done? (0 NO, 1 YES): ");
    std::cin >>done;
    if(done==1)
    {
    	break;
    }

}

	return 0;
}