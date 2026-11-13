"""Compiler-free reference propagation with the same per-path MT19937-64 stream."""
import numpy as np
from library import reservoir_scaling as physics

MASK=(1<<64)-1


def mix64(x):
    x=(x+0x9e3779b97f4a7c15)&MASK
    x=((x^(x>>30))*0xbf58476d1ce4e5b9)&MASK
    x=((x^(x>>27))*0x94d049bb133111eb)&MASK
    return (x^(x>>31))&MASK


class MT19937_64:
    """The engine parameters and seeding recurrence of C++ std::mt19937_64."""
    def __init__(self,seed):
        self.state=[seed&MASK]
        for i in range(1,312):self.state.append((6364136223846793005*(self.state[-1]^(self.state[-1]>>62))+i)&MASK)
        self.index=312

    def next(self):
        if self.index==312:
            for i in range(312):
                x=(self.state[i]&0xffffffff80000000)|(self.state[(i+1)%312]&0x7fffffff)
                self.state[i]=self.state[(i+156)%312]^(x>>1)^(0xb5026f5aa96619e9 if x&1 else 0)
            self.index=0
        x=self.state[self.index];self.index+=1
        x^=(x>>29)&0x5555555555555555
        x^=(x<<17)&0x71d67fffeda60000
        x^=(x<<37)&0xfff7eee000000000
        x^=x>>43
        return x&MASK


def references(case,R,seed,initial):
    n=case['N'];p=case['physics'];dts=case.get('step_dts',[case.get('constant_dt')]*case['steps'])
    model=physics.Model(n,p['gamma']/p['J'],p['p_left'],p['p_right'],p['prefactor'])
    unitaries=[physics.system_unitary(model,p['J']*dt,'even-odd') for dt in dts]
    bath=[np.cos(np.sqrt(2*p['gamma']*dt)) for dt in dts]
    mean=initial.copy()
    for dt in dts:mean=physics.mean_step(mean,physics.channel(model,p['J']*dt,'rate-weighted','hamiltonian-first','even-odd'))
    def obs(C):return np.r_[np.diag(C),4*p['prefactor']*np.diag(C,1)]
    choices=np.empty((R,len(dts)),dtype=np.int32);ideals=np.empty((R,2*n-1))
    for path in range(R):
        rng=MT19937_64(mix64(seed^mix64(path)));C=initial.copy()
        for k,(U,b) in enumerate(zip(unitaries,bath)):
            u=(rng.next()>>11)*2**-53
            choice=0 if u<p['p_left']/2 else 1 if u<.5 else 2 if u<.5+p['p_right']/2 else 3
            choices[path,k]=choice;C=U@C@U.T;site=0 if choice<2 else n-1
            C[site,:]*=b;C[:,site]*=b
            if choice%2==0:C[site,site]+=1-b*b
        ideals[path]=obs(C)
    return dict(choices=choices,ideal_paths=ideals,digital=obs(mean),target=obs(physics.exact_stationary(model)))
