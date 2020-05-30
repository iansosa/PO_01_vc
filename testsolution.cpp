#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <string>
#include <cassert>
#include <vector>
#include <iostream>

#include <complex>    

int sign(int i)
{
	if(i%2==0)
	{
		return 1;
	}
	if(i%2==1)
	{
		return -1;
	}
}

double calccosa_next(int N, double K,int N_capas,std::vector<double> &capas,std::vector<double> &capas_next,int icurrent,std::vector<double> &A,double gamma_med, double g,int caso,double x)
{
	double sum=0;
	for (int i = icurrent; i < N_capas; ++i)
	{
		sum=sum+capas[i]*pow(A[i],2)*(x*gamma_med+(g*K/N-1)*(K*(capas_next[i]+capas_next[i-1])/(capas[i]*N)-1)*sign(i-icurrent+caso));
	}
	return sum;
}

double calcN_next(int N_capas,std::vector<double> &capas,int icurrent)
{
	double sum=0;
	for (int i = icurrent; i < N_capas; ++i)
	{
		sum=sum+capas[i];
	}
	return(sum);
}

double calcA_next(int N_capas,std::vector<double> &capas,int icurrent,std::vector<double> &A)
{
	double sum=0;
	for (int i = icurrent; i < N_capas; ++i)
	{
		sum=sum+capas[i]*pow(A[i],2);
	}
	return(sum/calcN_next(N_capas,capas,icurrent));
}


void calcW_med(double w, double gamma_med, int N_capas,std::vector<double> &capas,std::vector<double> &A,std::vector<double> &W_prev,std::vector<double> &W_gamma,std::vector<double> &W_next)
{
	for (int i = 1; i < N_capas; ++i)
	{
		W_next[i]=-calcN_next(N_capas,capas,i+1)*(calcA_next(N_capas,capas,i+1,A)*gamma_med*0.5+gamma_med*w*w);
		W_next[i]=W_next[i]/capas[i];
		W_prev[i]=calcN_next(N_capas,capas,i)*(calcA_next(N_capas,capas,i,A)*gamma_med*0.5+gamma_med*w*w);
		W_prev[i]=W_prev[i]/capas[i];
		W_gamma[i]=-capas[i]*(pow(A[i],2)*gamma_med*0.5+gamma_med*w*w);
		W_gamma[i]=W_gamma[i]/capas[i];
	}

	for (int i = 1; i < N_capas; ++i)
	{
		printf("W_next[%d]=%.15lf\n",i, W_next[i]);
		printf("W_prev[%d]=%.15lf\n",i, W_prev[i]);
		printf("W_gamma[%d]=%.15lf\n",i, W_gamma[i]);
	}
}

double calcw(int N, int N_capas,std::vector<double> &capas,std::vector<double> &A)
{
	double w=0;

	for (int i = 1; i < N_capas; ++i)
	{
		w=w+capas[i]*pow(A[i],2);
	}
	w=w/2.0;
	w=w/(N-1);
	printf("w'=%.15lf\n", w);

	return(w);
}

void calcD(int N, double K,double gamma_med, int N_capas,std::vector<std::complex<double>> &D,std::vector<double> &capas,std::vector<double> &capas_next)
{
    using namespace std::complex_literals;

	int kappa=N_capas-1;
	D[0]=0; //no usar
	for (int i = 1; i <= kappa; ++i)
	{
		D[i]= (N/K)*((K/N)*((capas_next[i-1]+capas_next[i])/capas[i])-1)+1i*(N/K)*gamma_med;
	}
}

void calcd(int N, double K,double gamma_med, int N_capas,std::vector<std::complex<double>> &D,std::vector<double> &capas,std::vector<double> &capas_next,std::vector<double> &d,std::vector<double> &A)
{
	using namespace std::complex_literals;

	std::vector<double> aux_prod(N_capas);

	std::vector<std::complex<double>> aux_denom(N_capas);
	for (int i = 0; i < N_capas; ++i)
	{
		aux_denom[i]=0;
	}

	int kappa=N_capas-1;

	d[0]=0;//no usar
	aux_prod[kappa]=1;
	d[kappa]=capas_next[kappa-1]/(std::abs(D[kappa]*capas[kappa])); 

	std::complex<double> aux_re;
	std::complex<double> aux_im;
	for (int i = kappa-1; i > 0; --i)
	{
		aux_prod[i]=aux_prod[i+1]*d[i+1];
		for (int k = 0; k < N_capas; ++k)
		{
			aux_denom[k]=0;
		}
		for (int k = i; k <= kappa; ++k)
		{
			aux_re=sign(k)*real(D[k])*capas[k]*pow(aux_prod[kappa+i-k],2);
			aux_im=1i*(imag(D[k])*capas[k]*pow(aux_prod[kappa+i-k],2));
			aux_denom[k]=aux_denom[k]+aux_re;
			aux_denom[k]=aux_denom[k]+aux_im;
		}
		d[i]=capas_next[i-1]/std::abs(aux_denom[i]);		
	}

	aux_prod[0]=1;
	for (int i = 1; i < N_capas; ++i)
	{
		aux_prod[i]=aux_prod[i-1]*d[i];
	}

	A[0]=0;
	for (int i = 1; i < N_capas; ++i)
	{
		A[i]=aux_prod[i];
		printf("A[%d]=%.15lf\n",i,A[i] );
		printf("d[%d]=%lf\n",i,d[i] );
	}

}

void printWnext(int N, double K,int N_capas,std::vector<double> &capas,std::vector<double> &capas_next,int icurrent,std::vector<double> &A,double gamma_med, double g,FILE *f_prev,double w,double W_med,double gamma_min,double gamma_max)
{
	double dx=(gamma_max-gamma_min)/1000.0;
	double x;
	double delta_gamma;
	double Wcurrent;
	for (int i = 0; i < 1000; ++i)
	{
		x=i*dx+gamma_min;
		delta_gamma=x-gamma_med;
		Wcurrent=delta_gamma*capas_next[icurrent]*w*w/g+delta_gamma*0.5*calccosa_next(N,K,N_capas,capas,capas_next,icurrent+1,A,gamma_med,g,0,x)/(pow(K*g/N-1,2)+pow(x,2));
		fprintf(f_prev, "%lf   %.15lf\n",x,W_med+Wcurrent/capas[icurrent]);
	}
}

void printWprev(int N, double K,int N_capas,std::vector<double> &capas,std::vector<double> &capas_next,int icurrent,std::vector<double> &A,double gamma_med, double g,FILE *f_prev,double w,double W_med,double gamma_min,double gamma_max)
{
	double dx=(gamma_max-gamma_min)/1000.0;
	double x;
	double delta_gamma;
	double Wcurrent;
	for (int i = 0; i < 1000; ++i)
	{
		x=i*dx+gamma_min;
		delta_gamma=x-gamma_med;
		Wcurrent=delta_gamma*capas_next[icurrent-1]*w*w/g-delta_gamma*0.5*calccosa_next(N,K,N_capas,capas,capas_next,icurrent,A,gamma_med,g,1,x)/(pow(K*g/N-1,2)+pow(x,2));
		fprintf(f_prev, "%lf   %.15lf\n",x,W_med+Wcurrent/capas[icurrent]);
	}
}

void printWgamma(int N, double K,int N_capas,std::vector<double> &capas,std::vector<double> &capas_next,int icurrent,std::vector<double> &A,double gamma_med, double g,FILE *f_prev,double w,double W_med,double gamma_min,double gamma_max)
{
	double dx=(gamma_max-gamma_min)/1000.0;
	double x;
	double delta_gamma;
	double Wcurrent;
	for (int i = 0; i < 1000; ++i)
	{
		x=i*dx+gamma_min;
		delta_gamma=x-gamma_med;
		Wcurrent=-delta_gamma*capas[icurrent]*w*w-delta_gamma*capas[icurrent]*0.5*pow(A[icurrent],2)+delta_gamma*0.5*capas[icurrent]*pow(A[icurrent],2)*x*(gamma_med+x)/(pow(K*g/N-1,2)+pow(x,2));
		fprintf(f_prev, "%lf   %.15lf\n",x,W_med+Wcurrent/capas[icurrent]);
	}
}

void printWcurrent(int N, double K,int N_capas,std::vector<double> &capas,std::vector<double> &capas_next,int icurrent,std::vector<double> &A,double gamma_med, double g,FILE *f_prev,double w,double W_med,double gamma_min,double gamma_max)
{
	double dx=(gamma_max-gamma_min)/1000.0;
	double x;
	double delta_gamma;
	double Wcurrent;
	for (int i = 0; i < 1000; ++i)
	{
		x=i*dx+gamma_min;
		delta_gamma=x-gamma_med;
		Wcurrent=delta_gamma*(g*capas[icurrent]-capas_next[icurrent]-capas_next[icurrent-1])*w*w/g+delta_gamma*0.5*pow(A[icurrent],2)*(g*capas[icurrent]-capas_next[icurrent]-capas_next[icurrent-1])*(K*g/N-1)*(K/N)/(pow(K*g/N-1,2)+pow(x,2));
		fprintf(f_prev, "%lf   %.15lf\n",x,W_med+Wcurrent/capas[icurrent]);
	}
}

int main()
{
    using namespace std::complex_literals;

    int N;
    printf("N:  ");
    std::cin >>N;

    double K;
    printf("K:  ");
    std::cin >>K;

    int N_capas;
    printf("N_capas (contando la 0):  ");
    std::cin >>N_capas;

    std::vector<double> capas(N_capas);
    std::vector<double> capas_next(N_capas);
    for (int i = 0; i < N_capas; ++i)
    {
    	printf("N[%d] :  ",i);
    	std::cin >>capas[i];
    	printf("N_next[%d] :  ",i);
    	std::cin >>capas_next[i];
    }

    double g;
    printf("g (grado medio):  ");
    std::cin >>g;



    std::vector<double> capas_current(N_capas);
	for (int i = 0; i < N_capas; ++i)
	{
		if(i==0)
		{
			capas_current[i]=g*capas[i]-capas_next[i];
		}
		if(i>0)
		{
			capas_current[i]=g*capas[i]-capas_next[i]-capas_next[i-1];
		}
	    printf("N_current[%d] : %lf  \n",i,capas_current[i]);
	}

    double gamma_med;
    printf("<gamma>:  ");
    std::cin >>gamma_med;

    double g_min;
    printf("gamma_min:  ");
    std::cin >>g_min;

    double g_max;
    printf("gamma_max:  ");
    std::cin >>g_max;

	std::vector<std::complex<double>> D(N_capas); 
	calcD(N,K,gamma_med,N_capas,D,capas,capas_next);
    std::vector<double> d(N_capas);
    std::vector<double> A(N_capas);
    calcd(N,K,gamma_med,N_capas,D,capas,capas_next,d,A);

    double w=calcw(N,N_capas,capas,A);
    std::vector<double> W_prev(N_capas);
    std::vector<double> W_gamma(N_capas);
    std::vector<double> W_next(N_capas);
	calcW_med(w,gamma_med,N_capas,capas,A,W_prev,W_gamma,W_next);

	char savename_Wprev[80]={};
	char savename_Wcurrent[80]={};
	char savename_Wnext[80]={};
	char savename_Wgamma[80]={};

	FILE *f_prev;
	FILE *f_current;
	FILE *f_next;
	FILE *f_gamma;

	for (int i = 1; i < N_capas; ++i)
	{
    	sprintf(savename_Wprev,"test_Wprev_capa%d.txt",i);
		f_prev=fopen(savename_Wprev, "w");
		printWprev(N,K,N_capas,capas,capas_next,i,A,gamma_med,g,f_prev,w,W_prev[i],gamma_min,gamma_max);
		fclose(f_prev);

    	sprintf(savename_Wcurrent,"test_Wcurrent_capa%d.txt",i);
		f_current=fopen(savename_Wcurrent, "w");
		printWcurrent(N,K,N_capas,capas,capas_next,i,A,gamma_med,g,f_current,w,0,gamma_min,gamma_max);
		fclose(f_current);

    	sprintf(savename_Wnext,"test_Wnext_capa%d.txt",i);
		f_next=fopen(savename_Wnext, "w");
		printWnext(N,K,N_capas,capas,capas_next,i,A,gamma_med,g,f_next,w,W_next[i],gamma_min,gamma_max);
		fclose(f_next);

    	sprintf(savename_Wgamma,"test_Wgamma_capa%d.txt",i);
		f_gamma=fopen(savename_Wgamma, "w");
		printWgamma(N,K,N_capas,capas,capas_next,i,A,gamma_med,g,f_gamma,w,W_gamma[i],gamma_min,gamma_max);
		fclose(f_gamma);
	}

	return 0;
}