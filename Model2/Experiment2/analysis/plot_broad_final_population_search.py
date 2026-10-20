"""Plot the saved broader final-population parameter search."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


def main():
    output=Path(__file__).parent/'N20_broad_final_population_search'
    report=json.loads((output/'search.json').read_text())
    small=np.load(output/'dense_0_20.npz')
    medium=np.load(output/'dense_20_1000.npz')
    large=np.load(output/'sobol_log_1000_1e6.npz')
    best=report['best']
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    colors=axes[0].pcolormesh(small['J_tau'],small['cosines'],
        small['errors'].min(axis=2).T,shading='auto',norm=LogNorm(vmin=.07,vmax=.75),cmap='viridis_r')
    fig.colorbar(colors,ax=axes[0],label='Maximum final population error')
    ii,jj=np.unravel_index(small['errors'].min(axis=2).argmin(),small['errors'].shape[:2])
    axes[0].plot(small['J_tau'][ii],small['cosines'][jj],'r*',ms=10,label='Best on this grid')
    axes[0].set(xlabel=r'Hamiltonian rotation $J\tau$',ylabel=r'Reservoir cosine $\cos\sqrt{2\Gamma\tau}$',
        title='Dense scan; best of both factor orders')
    axes[0].legend(loc='upper right')
    for data,label in ((small,'Dense: 0–20'),(medium,'Dense: 20–1000')):
        axes[1].semilogx(data['J_tau'],data['errors'].min(axis=(1,2)),label=label,lw=.8)
    bins=np.geomspace(1000,1e6,101)
    groups=np.digitize(large['J_tau'],bins)-1
    ys=[]
    for i in range(len(bins)-1):
        ys.append(large['errors'][groups==i].min())
    axes[1].semilogx(np.sqrt(bins[:-1]*bins[1:]),ys,label='Best sampled per logarithmic bin',lw=1)
    axes[1].axhline(.001,color='black',ls='--',label='Required: 0.001')
    axes[1].axhline(best['error'],color='red',ls=':',label=f"Best found: {best['error']:.5f}")
    axes[1].plot(best['J_tau'],best['error'],'r*',ms=10)
    axes[1].set(xlabel=r'Hamiltonian rotation $J\tau$',ylabel='Final population error',
        title='Search through large rotation phases',ylim=(.0006,.85),yscale='log')
    axes[1].grid(alpha=.2)
    axes[1].legend(fontsize=8)
    fig.suptitle(r'$N=20$, 100 DLL resets, $p=1/4$, $\Gamma=20$, $J\leq100$')
    fig.tight_layout()
    fig.savefig(output/'parameter_search.png',dpi=180)
    fig.savefig(output/'parameter_search.pdf')
    report_path=output/'Results.md'
    original=report_path.read_text()
    if '![Parameter search]' not in original:
        report_path.write_text(original+'\n![Parameter search](parameter_search.png)\n')
    print(output/'parameter_search.png')


if __name__=='__main__':
    main()
