// Exact conditional covariance trajectories, matching Experiment1's channel.
// A center-excitation loss-only path has C=v*v^T and is evolved in O(N).
#include <cstdint>
#include <random>
#include <omp.h>
#include "../Experiment1/channel.cpp"

static void rotate_matrix(double* a, int n, double c, double s) {
    for (int parity=0; parity<2; ++parity) {
        for (int i=parity; i<n-1; i+=2) {
            double* x=a+i*n; double* y=x+n;
            #pragma omp simd
            for (int j=0; j<n; ++j) {
                const double u=x[j], v=y[j]; x[j]=c*u-s*v; y[j]=s*u+c*v;
            }
        }
        for (int i=0; i<n; ++i) {
            double* row=a+i*n;
            #pragma omp simd
            for (int j=parity; j<n-1; j+=2) {
                const double u=row[j], v=row[j+1]; row[j]=c*u-s*v; row[j+1]=s*u+c*v;
            }
        }
    }
}

static void path_step(double* a, int n, double c, double s, double b, int choice) {
    rotate_matrix(a,n,c,s);
    const int site=choice<2 ? 0 : n-1;
    for (int j=0; j<n; ++j) a[site*n+j]*=b;
    for (int i=0; i<n; ++i) a[i*n+site]*=b;
    if (choice%2==0) a[site*n+site]+=1-b*b;
}

static void vector_step(double* v, int n, double c, double s, double b, int choice) {
    for (int parity=0; parity<2; ++parity) {
        for (int i=parity; i<n-1; i+=2) {
            const double x=v[i], y=v[i+1]; v[i]=c*x-s*y; v[i+1]=s*x+c*y;
        }
    }
    v[choice<2 ? 0 : n-1]*=b;
}

static std::uint64_t mix64(std::uint64_t x) {
    x+=0x9e3779b97f4a7c15ULL;
    x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
    x=(x^(x>>27))*0x94d049bb133111ebULL;
    return x^(x>>31);
}

static int draw(std::mt19937_64& generator, double pl, double pr) {
    const double u=static_cast<double>(generator()>>11)*0x1.0p-53;
    return u<pl/2 ? 0 : u<.5 ? 1 : u<.5+pr/2 ? 2 : 3;
}

static double observable(const double* a, int n, int j, double factor, bool rank_one) {
    if (j<n) return rank_one ? a[j]*a[j] : a[j*n+j];
    const int i=j-n;
    return factor*(rank_one ? a[i]*a[i+1] : a[i*n+i+1]);
}

extern "C" void trajectory_reference(int n, double prefactor, double J, double gamma,
 double pl, double pr, const double* initial, const double* dts, int count, int tail,
 double* output) {
    std::vector<double> state(initial,initial+n*n);
    for (int k=0; k<count; ++k)
        step(state.data(),n,2*prefactor*J*dts[k],std::cos(std::sqrt(2*gamma*dts[k])),pl,pr);
    const int d=2*n-1;
    for (int t=0; t<=tail; ++t) {
        for (int j=0; j<d; ++j) output[t*d+j]=observable(state.data(),n,j,4*prefactor,false);
        if (t<tail) step(state.data(),n,2*prefactor*J*dts[count-1],
            std::cos(std::sqrt(2*gamma*dts[count-1])),pl,pr);
    }
}

extern "C" int trajectory_given_choices(int n, double prefactor, double J, double gamma,
 const double* initial, const double* dts, int count, int tail, const int* choices,
 int rank_one, double* state_output, double* observations) {
    const int d=2*n-1;
    std::vector<double> state(rank_one ? n : n*n,0.);
    if (rank_one) state[n/2]=1.; else std::copy(initial,initial+n*n,state.begin());
    for (int k=0; k<count+tail; ++k) {
        const double dt=dts[std::min(k,count-1)], angle=2*prefactor*J*dt;
        if (rank_one) vector_step(state.data(),n,std::cos(angle),std::sin(angle),
            std::cos(std::sqrt(2*gamma*dt)),choices[k]);
        else path_step(state.data(),n,std::cos(angle),std::sin(angle),
            std::cos(std::sqrt(2*gamma*dt)),choices[k]);
        if (k>=count-1) {
            const int t=k-count+1;
            for (int j=0; j<d; ++j) observations[t*d+j]=observable(state.data(),n,j,4*prefactor,rank_one);
        }
    }
    for (int i=0; i<n; ++i) for (int j=0; j<n; ++j)
        state_output[i*n+j]=rank_one ? state[i]*state[j] : state[i*n+j];
    return 0;
}

extern "C" int trajectory_batch(int n, double prefactor, double J, double gamma,
 double pl, double pr, const double* initial, const double* dts, int count, int tail,
 const double* reference, std::uint64_t seed, std::uint64_t first_path, int paths,
 int threads, int rank_one, double* sums, double* squares, double* bounds) {
    const int d=2*n-1, total=count+tail;
    const std::size_t size=static_cast<std::size_t>(tail+1)*d;
    std::vector<double> cosines(total),sines(total),bath(total);
    for (int k=0; k<total; ++k) {
        const double dt=dts[std::min(k,count-1)];
        cosines[k]=std::cos(2*prefactor*J*dt);sines[k]=std::sin(2*prefactor*J*dt);
        bath[k]=std::cos(std::sqrt(2*gamma*dt));
    }
    std::vector<std::vector<double>> local_sums(threads,std::vector<double>(size,0.));
    std::vector<std::vector<double>> local_squares(threads,std::vector<double>(size,0.));
    std::vector<double> minimum(threads,1.),maximum(threads,0.),current(threads,0.);
    #pragma omp parallel num_threads(threads)
    {
        const int thread=omp_get_thread_num();
        double* sum=local_sums[thread].data();double* square=local_squares[thread].data();
        std::vector<double> state(rank_one ? n : n*n);
        #pragma omp for schedule(static)
        for (int path=0; path<paths; ++path) {
            std::fill(state.begin(),state.end(),0.);
            if (rank_one) state[n/2]=1.; else std::copy(initial,initial+n*n,state.begin());
            std::mt19937_64 generator(mix64(seed^mix64(first_path+path)));
            for (int k=0; k<total; ++k) {
                const int choice=draw(generator,pl,pr);
                if (rank_one) vector_step(state.data(),n,cosines[k],sines[k],bath[k],choice);
                else path_step(state.data(),n,cosines[k],sines[k],bath[k],choice);
                if (k>=count-1) {
                    const std::size_t start=static_cast<std::size_t>(k-count+1)*d;
                    for (int j=0; j<d; ++j) {
                        const double value=observable(state.data(),n,j,4*prefactor,rank_one);
                        const double delta=value-reference[start+j];
                        sum[start+j]+=delta;square[start+j]+=delta*delta;
                        if (j<n) {minimum[thread]=std::min(minimum[thread],value);maximum[thread]=std::max(maximum[thread],value);}
                        else current[thread]=std::max(current[thread],std::abs(value));
                    }
                }
            }
        }
    }
    std::fill(sums,sums+size,0.);std::fill(squares,squares+size,0.);
    for (int thread=0; thread<threads; ++thread) for (std::size_t j=0; j<size; ++j) {
        sums[j]+=local_sums[thread][j];squares[j]+=local_squares[thread][j];
    }
    bounds[0]=*std::min_element(minimum.begin(),minimum.end());
    bounds[1]=*std::max_element(maximum.begin(),maximum.end());
    bounds[2]=*std::max_element(current.begin(),current.end());
    const double current_limit=2*prefactor;
    return bounds[0]<-1e-8 || bounds[1]>1+1e-8 || bounds[2]>current_limit+1e-8 ? -1 : 0;
}
