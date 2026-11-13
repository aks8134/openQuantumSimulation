// Correlated Gaussian projective readout: no statevector or circuit simulator.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
#include <omp.h>
#include "../Experiment2/trajectory.cpp"

static std::uint64_t shot_mix64(std::uint64_t x) {
    x+=0x9e3779b97f4a7c15ULL;
    x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
    x=(x^(x>>27))*0x94d049bb133111ebULL;
    return x^(x>>31);
}

extern "C" int gaussian_sample(const double* input, int d, int pairs, int shots,
 std::uint64_t seed, std::uint64_t first, int threads, signed char* output) {
    int invalid=0;
    #pragma omp parallel num_threads(threads) reduction(|:invalid)
    {
        std::vector<double> a(d*d);
        // Fixed 64-shot blocks amortize RNG initialization while keeping prefixes
        // bit-identical across team sizes and arbitrary batch boundaries.
        const std::uint64_t block_size=64;
        const std::uint64_t first_block=first/block_size;
        const std::uint64_t last_block=(first+shots-1)/block_size;
        #pragma omp for schedule(static)
        for (std::uint64_t block=first_block; block<=last_block; ++block) {
            std::mt19937_64 rng(shot_mix64(seed^shot_mix64(block)));
            const std::uint64_t begin=std::max(first,block*block_size);
            const std::uint64_t end=std::min(first+shots,(block+1)*block_size);
            rng.discard((begin-block*block_size)*pairs);
            for (std::uint64_t index=begin; index<end; ++index) {
            const int shot=static_cast<int>(index-first);
            std::copy(input,input+d*d,a.begin());
            for (int k=0; k<pairs; ++k) {
                const int p=2*k, q=p+1;
                double g=a[p*d+q];
                if (!std::isfinite(g) || std::abs(g)>1+1e-8) {invalid=1;break;}
                g=std::max(-1.,std::min(1.,g));
                const double u=static_cast<double>(rng()>>11)*0x1.0p-53;
                const int sign=u<(1+g)/2 ? 1 : -1;
                output[shot*pairs+k]=static_cast<signed char>(sign);
                const double weight=sign/(1+sign*g);
                const double* rowp=a.data()+p*d;
                const double* rowq=a.data()+q*d;
                for (int i=q+1; i<d; ++i) {
                    double* row=a.data()+i*d;
                    const double pi=rowp[i], qi=rowq[i];
                    #pragma omp simd
                    for (int j=i+1; j<d; ++j)
                        row[j]+=weight*(qi*rowp[j]-pi*rowq[j]);
                }
            }
            }
        }
    }
    return invalid;
}

extern "C" void trajectory_snapshot(int n, double prefactor, double J, double gamma,
 double pl, double pr, const double* initial, const double* dts, int count, int tail,
 std::uint64_t seed, std::uint64_t path_index, int rank_one, double* covariance, int* choices) {
    std::vector<double> state(rank_one ? n : n*n,0.);
    if (rank_one) state[n/2]=1.; else std::copy(initial,initial+n*n,state.begin());
    std::mt19937_64 rng(mix64(seed^mix64(path_index)));
    for (int k=0; k<count+tail; ++k) {
        const double dt=dts[std::min(k,count-1)], angle=2*prefactor*J*dt;
        const int choice=draw(rng,pl,pr);
        choices[k]=choice;
        if (rank_one) vector_step(state.data(),n,std::cos(angle),std::sin(angle),
                                 std::cos(std::sqrt(2*gamma*dt)),choice);
        else path_step(state.data(),n,std::cos(angle),std::sin(angle),
                       std::cos(std::sqrt(2*gamma*dt)),choice);
    }
    for (int i=0; i<n; ++i) for (int j=0; j<n; ++j)
        covariance[i*n+j]=rank_one ? state[i]*state[j] : state[i*n+j];
}
