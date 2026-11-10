// Real-gauge, deterministic mean XX channel. No quantum state simulator.
#include <algorithm>
#include <cmath>
#include <vector>

static void step(double* a, int n, double angle, double bath, double pl, double pr) {
    const double c=std::cos(angle), s=std::sin(angle);
    for (int parity=0; parity<2; ++parity) {
        for (int i=parity; i<n-1; i+=2) {
            double* x=a+i*n; double* y=x+n;
            for (int j=0; j<n; ++j) {
                const double u=x[j], v=y[j]; x[j]=c*u-s*v; y[j]=s*u+c*v;
            }
        }
        for (int i=0; i<n; ++i) {
            double* row=a+i*n;
            for (int j=parity; j<n-1; j+=2) {
                const double u=row[j], v=row[j+1]; row[j]=c*u-s*v; row[j+1]=s*u+c*v;
            }
        }
    }
    const double f=(1+bath)/2, d=(1+bath*bath)/2, b=(1-bath*bath)/2;
    for (int j=1; j<n-1; ++j) {
        a[j]*=f; a[j*n]*=f; a[(n-1)*n+j]*=f; a[j*n+n-1]*=f;
    }
    a[0]=d*a[0]+pl*b; a[n*n-1]=d*a[n*n-1]+pr*b;
    a[n-1]*=bath; a[(n-1)*n]*=bath;
}

static double error(const double* a, const double* target, int n, double factor) {
    double result=0;
    for (int i=0; i<n; ++i) result=std::max(result,std::abs(a[i*n+i]-target[i*n+i]));
    for (int i=0; i<n-1; ++i) result=std::max(result,factor*std::abs(a[i*n+i+1]-target[i*n+i+1]));
    return result;
}

extern "C" int evolve(int n, double prefactor, double J, double gamma, double pl, double pr,
 const double* initial, const double* target, const double* dts, int count,
 double tolerance, double window, int validation_cap, double* output, double* stats) {
    std::vector<double> a(initial,initial+n*n);
    for (int k=0; k<count; ++k) step(a.data(),n,2*prefactor*J*dts[k],std::cos(std::sqrt(2*gamma*dts[k])),pl,pr);
    std::copy(a.begin(),a.end(),output);
    double maximum=error(a.data(),target,n,4*prefactor);
    const int tail=static_cast<int>(std::ceil(window/dts[count-1]));
    stats[0]=maximum; stats[2]=tail; stats[3]=0;
    if (maximum<=tolerance && tail<=validation_cap) {
        const double angle=2*prefactor*J*dts[count-1], bath=std::cos(std::sqrt(2*gamma*dts[count-1]));
        for (int k=0; k<tail; ++k) {
            step(a.data(),n,angle,bath,pl,pr);
            maximum=std::max(maximum,error(a.data(),target,n,4*prefactor));
            stats[3]=k+1;
            if (maximum>tolerance) break;
        }
    }
    stats[1]=maximum;
    return maximum<=tolerance && tail<=validation_cap;
}

extern "C" int constant_crossing(int n, double prefactor, double J, double gamma, double pl, double pr,
 const double* initial, const double* target, double dt, int cap,
 double tolerance, double window, int validation_cap) {
    std::vector<double> a(initial,initial+n*n);
    const double angle=2*prefactor*J*dt, bath=std::cos(std::sqrt(2*gamma*dt));
    const int tail=static_cast<int>(std::ceil(window/dt));
    if (tail>validation_cap) return -1;
    for (int k=1; k<=cap; ++k) {
        step(a.data(),n,angle,bath,pl,pr);
        if (error(a.data(),target,n,4*prefactor)>tolerance) continue;
        std::vector<double> future=a;
        bool passed=true;
        for (int j=0; j<tail; ++j) {
            step(future.data(),n,angle,bath,pl,pr);
            if (error(future.data(),target,n,4*prefactor)>tolerance) { passed=false; break; }
        }
        if (passed) return k;
    }
    return -1;
}
