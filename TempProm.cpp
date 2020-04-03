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

double c_1b(double K, double r_up, double gamma_down)
{
    return(K*r_up/(sqrt(pow(K*r_up-1,2)+pow(gamma_down,2))));
}

double Aup_1b(double K,double r_up,double r_down,double gamma_up,double gamma_down,int N)
{
    double c=c_1b(K,r_up,gamma_down);
    return((K/N)*(1.0/sqrt(pow(K*r_down-1-c*c*(r_down/r_up)*(K*r_up-1),2)+pow(gamma_up+gamma_down*(r_down/r_up)*c*c,2))));
}

double Adown_1b(double K,double r_up,double r_down,double gamma_up,double gamma_down,int N)
{
    double c=c_1b(K,r_up,gamma_down);
    return(c*Aup_1b(K,r_up,r_down,gamma_up,gamma_down,N));
}

double Te1_up_1b(double K,double r_up,double r_down,double gamma_up,double gamma_down,int N,double x)
{
    double A_up=Aup_1b(K,r_up,r_down,gamma_up,gamma_down,N);
    double c=c_1b(K,r_up,gamma_down);

    double multiconst=A_up*A_up*0.5;
    double term1=gamma_up+(r_down/r_up)*gamma_down*c*c;
    double term2=(x-gamma_up)*(1.0/(pow(K*r_up-1,2)+pow(x,2)))*((K*r_down-1-c*c*(r_down/r_up)*(K*r_up-1))*(K*r_up-1)-x*(gamma_up+(r_down/r_up)*gamma_down*c*c));
    return(multiconst*(term1+term2));

}

double Tomega_up_1b(double K,double r_up,double r_down,double gamma_up,double gamma_down,int N,double x)
{
    double A_up=Aup_1b(K,r_up,r_down,gamma_up,gamma_down,N);
    double c=c_1b(K,r_up,gamma_down);
    double A_down=Adown_1b(K,r_up,r_down,gamma_up,gamma_down,N);

    double multiconst=A_up*A_up*0.5;
    double term1=-c*c*gamma_down*(r_down/r_up);
    double term2=(x-gamma_up)*(1.0/(pow(K*r_up-1,2)+pow(x,2)))*(K*r_up*(K*r_up-1)+c*c*(r_down/r_up)*(pow(K*r_up-1,2)+x*gamma_down)-r_down*K*(K*r_up-1));
    return(multiconst*(term1+term2));
}

double Tomega_down_1b(double K,double r_up,double r_down,double gamma_up,double gamma_down,int N,double x)
{
    double A_up=Aup_1b(K,r_up,r_down,gamma_up,gamma_down,N);
    double c=c_1b(K,r_up,gamma_down);
    double A_down=Adown_1b(K,r_up,r_down,gamma_up,gamma_down,N);

    double multiconst=A_down*A_down*0.5;
    double term1=gamma_down;
    double term2=(x-gamma_down)*(1.0/(pow(K*r_down-1,2)+pow(x,2)))*(K*r_down*(K*r_down-1)+(K*r_down-1)*(K*r_up-1)-x*gamma_down-r_up*K*(K*r_down-1));
    return(multiconst*(term1+term2));
}

double Tgamma_up_1b(double K,double r_up,double r_down,double gamma_up,double gamma_down,int N,double x)
{
    double A_up=Aup_1b(K,r_up,r_down,gamma_up,gamma_down,N);
    double c=c_1b(K,r_up,gamma_down);
    double A_down=Adown_1b(K,r_up,r_down,gamma_up,gamma_down,N);

    double multiconst=-x*A_up*A_up*0.5;
    double term1=1.0;
    double term2=-(x-gamma_up)*(1.0/(pow(K*r_up-1,2)+pow(x,2)))*(x+gamma_up);
    return(multiconst*(term1+term2));
}

double Tgamma_down_1b(double K,double r_up,double r_down,double gamma_up,double gamma_down,int N,double x)
{
    double A_up=Aup_1b(K,r_up,r_down,gamma_up,gamma_down,N);
    double c=c_1b(K,r_up,gamma_down);
    double A_down=Adown_1b(K,r_up,r_down,gamma_up,gamma_down,N);

    double multiconst=-x*A_down*A_down*0.5;
    double term1=1.0;
    double term2=-(x-gamma_down)*(1.0/(pow(K*r_down-1,2)+pow(x,2)))*(x+gamma_down);
    return(multiconst*(term1+term2));
}

void fillG(std::vector<double> &G,int N)
{
	FILE *r= fopen("Gi.txt", "r");
	for (int i = 0; i < N; ++i)
	{
		fscanf(r, "%lf", &G[i]);
	}
	fclose(r);
}

void fillI(std::vector<double> &I,int N)
{
	FILE *r= fopen("Ii.txt", "r");
	for (int i = 0; i < N; ++i)
	{
		fscanf(r, "%lf", &I[i]);
	}
	fclose(r);
}

void fillA(arma::Mat<double> &A,int N)
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

int calcsteps(int N, int n_p, int n_t)
{
    FILE* f = fopen("ac.txt", "r");
    int l=0;
    double input;
    while(1==1)
    {
        if(fscanf(f, "%lf", &input)==EOF)
        {
            fclose(f);
            printf("done\n");
            break;
        }
    	for (int i = 0; i < n_p*N; ++i)
    	{
            fscanf(f, "%lf", &input);
    	}
        for (int i = 0; i < n_t; ++i)
        {
            if(fscanf(f, "%lf", &input)==EOF)
            {
            	fclose(f);
                printf("doneT\n");
                break;
            }
        }
        l=l+1;
    }
    return(l);
}

void prinstuff_1b(int N,std::vector<double> &G,arma::Mat<double> &A,std::vector<double> &I,std::vector<double> &e1,std::vector<double> &eomega,std::vector<double> &egamma)
{
	FILE *f0=fopen("TempProm.txt","w");

    double K;
    printf("K: ");
    std::cin >>K;

    double r_up;
    printf("r+: ");
    std::cin >>r_up;

    double r_down;
    printf("r-: ");
    std::cin >>r_down;

    double gamma_up;
    printf("<G>+: ");
    std::cin >>gamma_up;

    double gamma_down;
    printf("<G>-: ");
    std::cin >>gamma_down;

	for (int i = 1; i < N; ++i)
	{
		fprintf(f0,"%.15lf   %.15lf   %.15lf   %.15lf	  %.15lf    %.15lf    %.15lf    %.15lf    %.15lf    %.15lf    %.15lf   \n",G[i]/I[i],A(i,0)/I[i],e1[i],eomega[i],egamma[i],Te1_up_1b(K,r_up,r_down,gamma_up,gamma_down,N,G[i]/I[i]),0.0,Tomega_up_1b(K,r_up,r_down,gamma_up,gamma_down,N,G[i]/I[i]),Tomega_down_1b(K,r_up,r_down,gamma_up,gamma_down,N,G[i]/I[i]),Tgamma_up_1b(K,r_up,r_down,gamma_up,gamma_down,N,G[i]/I[i]),Tgamma_down_1b(K,r_up,r_down,gamma_up,gamma_down,N,G[i]/I[i]));
        //fprintf(f0, " \n" );
	}
	fclose(f0);
    printf("A+=%lf\n",Aup_1b(K,r_up,r_down,gamma_up,gamma_down,N));
    printf("A-=%lf\n",Adown_1b(K,r_up,r_down,gamma_up,gamma_down,N));
}

int main()
{
	using namespace std;
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    double input;
    int N;
    printf("N: ");
    std::cin >>N;
    int n_p;
    printf("propiedades pp (x,p,epo,ecin,etot,e1,etransf,elost): ");
    std::cin >>n_p;
    int n_t;
    printf("propiedades T (0): ");
    std::cin >>n_t;
    double activestepsin;
	printf("active steps interval in [0:1]: ");
    std::cin >>activestepsin;
    double activestepsout;
	printf("active steps interval out [0:1]: ");
    std::cin >>activestepsout;


 	vector<double> e1(N);
	std::fill(e1.begin(), e1.end(), 0);
 	vector<double> eomega(N);
	std::fill(eomega.begin(), eomega.end(), 0);
 	vector<double> egamma(N);
	std::fill(egamma.begin(), egamma.end(), 0);

	vector<double> G(N);
	vector<double> I(N);
    arma::Mat<double> A(N,N);
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	int steps=calcsteps(N,n_p,n_t);

	int astepsin=(int) (steps*activestepsin);
	int astepsout=(int) (steps*activestepsout);
    printf("steps: %d\n",steps);
    printf("active stepsin: %d\n",astepsin);
    printf("active stepsout: %d\n",astepsout);
    vector<double> times(steps);
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    FILE *f = fopen("ac.txt", "r");
    int l=0;
  	for (int j = 0; j < steps; ++j)
  	{
        if(fscanf(f, "%lf", &input)==EOF)
        {
            fclose(f);
            printf("done: %d\n",j);
            break;
        }
        times[j]=input;
        l=0;
    	for (int i = 0; i < n_p*N; ++i)
    	{
            fscanf(f, "%lf", &input);
            if(j>=astepsin && j<=astepsout)
            {
    			if(i%n_p==5)
    			{
    				e1[l]=e1[l]+input;
    			}
       			if(i%n_p==6)
    			{
    				eomega[l]=eomega[l]+input;
    			}
       			if(i%n_p==7)
    			{
    				egamma[l]=egamma[l]+input;
    				l=l+1;
    			}
    		}
    	}
        for (int i = 0; i < n_t; ++i)
        {
            if(fscanf(f, "%lf", &input)==EOF)
            {
                fclose(f);
                printf("doneT\n");
                break;
            }
        }
    }

	fillG(G,N);
	fillI(I,N);
	fillA(A,N);
    for (int i = 0; i < N; ++i)
	{
    	e1[i]=(double) (e1[i]/(astepsout-astepsin));
        eomega[i]=(double) (eomega[i]/(astepsout-astepsin));
        egamma[i]=(double) (egamma[i]/(astepsout-astepsin));
	}
	printf("stuff loaded\n");
    //////////////////////////////////////////////////////////////////////////////////////////////////////////
   
	prinstuff_1b(N,G,A,I,e1,eomega,egamma);
	return 0;
}