"""Pure circuit/statistical helpers for first-order Model3 mitigation.

No service construction, hardware queries or submission on import.
"""
from __future__ import annotations

import numpy as np
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGatesDecomposition

from Model3.Experiment5.analyze_hardware import moments


def corrected_population(population, zero, one, min_contrast=.5):
    """Diagnostic marginal SPAM inversion, without clipping or target fitting."""
    population,zero,one=map(np.asarray,(population,zero,one))
    contrast=one-zero
    if not np.all(np.isfinite(contrast)) or np.any(contrast<min_contrast):
        raise ValueError('zero/one response has insufficient contrast for stable diagnostic inversion')
    return (population-zero)/contrast


def response_covariance(population,covariance,zero,zero_covariance,one,one_covariance):
    """Delta-method covariance, including independent shared calibration data."""
    corrected=corrected_population(population,zero,one);d=one-zero
    derivatives=(1/d,(corrected-1)/d,-corrected/d)
    return sum(v[:,None]*cov*v[None,:] for v,cov in zip(derivatives,
        (covariance,zero_covariance,one_covariance)))


def zne_weights(factors):
    """Predeclared unweighted linear intercept; no fit/model selection on targets."""
    x=np.asarray(factors,float)
    if x.ndim!=1 or len(x)<3 or len(np.unique(x))!=len(x) or np.any(~np.isfinite(x)) or min(x)<1:
        raise ValueError('ZNE requires at least three distinct finite factors >=1')
    design=np.column_stack([np.ones(len(x)),x])
    weights=np.linalg.pinv(design)[0]
    np.testing.assert_allclose(weights@np.ones(len(x)),1,atol=1e-12)
    np.testing.assert_allclose(weights@x,0,atol=1e-12)
    return weights


def balanced_masks(n,count,seed):
    if n<1 or count<2 or count%2:raise ValueError('measurement twirls require an even number >=2')
    rng=np.random.default_rng(seed);masks=[]
    for _ in range(count//2):
        bits=rng.integers(0,2,size=n);value=sum(int(b)<<j for j,b in enumerate(bits))
        masks.extend([value,value^((1<<n)-1)])
    return masks


def undo_measurement_twirl(counts,n,mask):
    if not 0<=mask<2**n:raise ValueError('invalid measurement-twirl mask')
    moments(counts,n)
    return {format(int(label.replace(' ',''),2)^mask,f'0{n}b'):count for label,count in counts.items()}


def fold_count(factor,cz_count):
    if not np.isfinite(factor) or factor<1 or cz_count<1:raise ValueError('invalid CZ folding request')
    count=int(np.floor((factor-1)*cz_count/2+.5))
    if count>cz_count:raise ValueError('selective triple-CZ folding supports noise factors up to 3')
    return count


def pauli(circuit,q,x,z):
    # RZ(pi) is Z up to global phase and is already native.
    if x:circuit.x(q)
    if z:circuit.rz(np.pi,q)


def twirl_cz(circuit,pair,bits,*,barrier_before=False):
    """Q CZ P = CZ up to global phase; CZ conjugation fixes Q from P."""
    x0,z0,x1,z1=map(int,bits);a,b=pair
    pauli(circuit,a,x0,z0);pauli(circuit,b,x1,z1)
    if barrier_before:circuit.barrier(a,b)
    circuit.cz(a,b)
    pauli(circuit,a,x0,z0^x1);pauli(circuit,b,x1,z1^x0)


def transformed(native,*,seed,folds,measurement_mask):
    """Fold/twirl native CZ only; never transform preparation or reset channels.

    The first `folds` elements of one seeded permutation are folded, making
    subsets nested across noise levels. Frames for all three possible copies
    are generated in advance, so common copies retain common frames.
    A restricted one-qubit pass absorbs frames without optimizing CZ/reset
    operations. Do not run a general transpiler afterward. Barriers preserve
    inserted CZ copies and separate terminal flips from preparation gates.
    """
    n=len(native.metadata['physical_sites']);cz=int(native.count_ops().get('cz',0))
    if folds<0 or folds>cz or not 0<=measurement_mask<2**n:raise ValueError('invalid transform')
    rng=np.random.default_rng(seed);selected=set(rng.permutation(cz)[:folds].tolist())
    frames=rng.integers(0,2,size=(cz,3,4));output=native.copy_empty_like()
    outcome=next(r for r in native.cregs if r.name=='outcome');index=0
    for item in native.data:
        op=item.operation
        if op.name in ('if_else','while_loop','switch_case') or getattr(op,'condition',None) is not None:
            raise ValueError('conditional operations require a separately validated transform')
        if op.name=='cz':
            for copy in range(3 if index in selected else 1):
                # Put the boundary after the pre-frame: adjacent post/pre
                # frames can simplify, but folded CZ copies cannot cancel.
                twirl_cz(output,item.qubits,frames[index,copy],barrier_before=bool(copy))
            index+=1
        else:
            if op.name=='measure' and item.clbits[0] in outcome:
                j=list(outcome).index(item.clbits[0])
                output.barrier(item.qubits[0])
                if (measurement_mask>>j)&1:output.x(item.qubits[0])
            output.append(op,item.qubits,item.clbits)
    output=PassManager([Optimize1qGatesDecomposition(basis=['rz','sx','x'])]).run(output)
    output.metadata={**(native.metadata or {}),'mitigation':dict(seed=int(seed),fold_count=folds,
        base_cz=cz,actual_factor=1+2*folds/cz if cz else 1.,measurement_mask=measurement_mask,
        folded_cz_indices=sorted(selected),manual_gate_twirling=True,one_qubit_frame_optimization=True)}
    if output.count_ops().get('cz',0)!=cz+2*folds:raise ValueError('CZ fold count changed')
    if output.count_ops().get('reset',0)!=native.count_ops().get('reset',0):raise ValueError('reset count changed')
    if output.count_ops().get('measure',0)!=native.count_ops().get('measure',0):raise ValueError('measurement count changed')
    return output


def group_moments(groups,n):
    """Equal-weight randomization means; groups need equal shots within an arm."""
    measured=[moments(c,n) for c in groups]
    if len({v[2] for v in measured})!=1:raise ValueError('unequal randomization shots')
    k=len(groups)
    return np.mean([v[0] for v in measured],axis=0),sum(v[1] for v in measured)/k**2


def draw_control(groups,n,replicates,rng):
    """Joint bitstring bootstrap within each calibration PUB, preserving contexts."""
    output=np.zeros((replicates,n))
    for counts in groups:
        labels=list(counts);weights=np.array([counts[key] for key in labels]);shots=int(weights.sum())
        bits=np.array([[(int(key,2)>>j)&1 for j in range(n)] for key in labels])
        output+=rng.multinomial(shots,weights/shots,size=replicates)@bits/shots/len(groups)
    return output


def estimates(values,shot_covariances,controls,factors,ideal_paths,target,*,replicates=2000,seed=29,groups=None):
    """Joint paired-path and calibration bootstrap, not double-resampled shots.

    `values` is (paths, arms, sites), raw first and twirled factors next.
    Each empirical path mean already includes measurement fluctuations.
    Calibration groups are resampled independently, once per replicate and
    shared across all factors. Variants/controls are never fit to the targets.
    """
    values=np.asarray(values,float);shot_covariances=np.asarray(shot_covariances,float)
    ideal_paths=np.asarray(ideal_paths,float);R,arms,n=values.shape
    if R<2 or arms!=len(factors)+1 or ideal_paths.shape!=(R,n) or replicates<100:
        raise ValueError('inconsistent mitigation analysis shapes or bootstrap count')
    w=zne_weights(factors);mean=values.mean(axis=0)
    cal={arm:{name:group_moments(groups,n) for name,groups in group.items()} for arm,group in controls.items()}
    def profiles(m,a_raw,b_raw,a_twirl,b_twirl):
        raw_c=corrected_population(m[...,0,:],a_raw,b_raw)
        twirl_c=corrected_population(m[...,1:,:],np.expand_dims(a_twirl,-2),np.expand_dims(b_twirl,-2))
        zne=np.einsum('k,...kj->...j',w,m[...,1:,:])
        zne_c=corrected_population(zne,a_twirl,b_twirl)
        return {'raw':m[...,0,:],'raw_spam_corrected':raw_c,
            'twirled':m[...,1,:],'twirled_spam_corrected':twirl_c[...,0,:],
            'zne':zne,'zne_spam_corrected':zne_c}
    points=profiles(mean,*[cal[arm][name][0] for arm in ('raw','twirled') for name in ('all_zero','all_one')])
    rng=np.random.default_rng(seed)
    cal_draws={arm:{name:draw_control(groups,n,replicates,rng) for name,groups in group.items()} for arm,group in controls.items()}
    # A noninvertible resampled calibration is recorded as a failure, not clipped.
    if groups is None:groups=[np.arange(R)]
    groups=[np.asarray(g,dtype=int) for g in groups if len(g)]
    if sorted(np.concatenate(groups).tolist())!=list(range(R)):raise ValueError('bootstrap groups must partition the trajectory indices')
    weights=np.zeros((replicates,R))
    for indices in groups:
        size=len(indices);weights[:,indices]=rng.multinomial(size,np.full(size,1/size),size=replicates)/R
    sampled=np.einsum('br,rkj->bkj',weights,values)
    samples=profiles(sampled,*[cal_draws[arm][name] for arm in ('raw','twirled') for name in ('all_zero','all_one')])
    ideal=ideal_paths.mean(axis=0);bootstrap_ideal=weights@ideal_paths
    output={};conditional={}
    raw_cov=shot_covariances.sum(axis=0)/R**2
    zne_cov=np.einsum('k,kij->ij',w*w,raw_cov[1:])
    covariance_map={'raw':raw_cov[0],'twirled':raw_cov[1],'zne':zne_cov}
    for name,cov in list(covariance_map.items()):
        arm='raw' if name=='raw' else 'twirled';a,ca=cal[arm]['all_zero'];b,cb=cal[arm]['all_one']
        covariance_map[name+'_spam_corrected']=response_covariance(points[name],cov,a,ca,b,cb)
    for name,point in points.items():
        error=abs(point-ideal);physical=abs(point-target)
        conditional[name]=np.sqrt(np.maximum(0,np.diag(covariance_map[name])))
        output[name]=dict(populations=point,bootstrap_ci95=np.quantile(samples[name],[.025,.975],axis=0),
            fixed_path_delta_method_se=conditional[name],outside_probability_range=bool(np.any((point<0)|(point>1))),
            ideal_mae=float(error.mean()),ideal_maximum=float(error.max()),physical_mae=float(physical.mean()),physical_maximum=float(physical.max()))
    comparisons=[]
    for name,reference in [('raw_spam_corrected','raw'),('twirled','raw'),('zne','twirled'),
                            ('twirled_spam_corrected','raw_spam_corrected'),
                            ('zne_spam_corrected','twirled_spam_corrected'),('zne_spam_corrected','raw')]:
        for metric in ('mae','maximum'):
            f=np.mean if metric=='mae' else np.max
            delta=f(abs(samples[name]-bootstrap_ideal),axis=1)-f(abs(samples[reference]-bootstrap_ideal),axis=1)
            comparisons.append(dict(method=name,reference=reference,metric='ideal_'+metric,
                difference=output[name]['ideal_'+metric]-output[reference]['ideal_'+metric],
                bootstrap_ci95=np.quantile(delta,[.025,.975])))
    # Linearity diagnostics are descriptive; correlated shared calibration is not
    # used as an independent-point chi-square goodness-of-fit test.
    design=np.column_stack([np.ones(len(factors)),factors]);fit=design@np.linalg.lstsq(design,mean[1:],rcond=None)[0]
    return dict(profiles=output,comparisons=comparisons,linear_zne=dict(factors=factors,weights=w,
        intercept_shot_se_amplification=float(np.linalg.norm(w)),raw_scale_populations=mean[1:],
        raw_scale_fixed_path_se=np.sqrt(np.maximum(0,np.diagonal(raw_cov[1:],axis1=-2,axis2=-1))),
        linear_fit_populations=fit,linear_fit_residuals=mean[1:]-fit),
        calibration={arm:{name:dict(populations=v[0],covariance=v[1]) for name,v in group.items()} for arm,group in cal.items()},
        heldout_mixture={arm:dict(raw=cal[arm]['initial_mixture'][0],
            corrected=corrected_population(cal[arm]['initial_mixture'][0],cal[arm]['all_zero'][0],cal[arm]['all_one'][0]),
            expected=.5,bootstrap_ci95=np.quantile(corrected_population(cal_draws[arm]['initial_mixture'],
                cal_draws[arm]['all_zero'],cal_draws[arm]['all_one']),[.025,.975],axis=0)) for arm in cal},
        bootstrap=dict(replicates=replicates,seed=seed,group_sizes=[len(g) for g in groups],
            scope='Paired empirical trajectory resampling within observed batches plus joint control-PUB resampling; shot noise already present in path means is not resampled again. Conditional on observed jobs and twirl variants; omits drift and calibration-model bias. Intervals descriptive, not simultaneous. Small-group bootstrap precision can be poor; fixed-path delta-method SEs are also reported.'),
        scope='Marginal SPAM model is conditional on stable local response and includes preparation error. CZ-only extrapolation does not remove intentional reservoir dissipation, finite-step bias, or guarantee removal of reset/idle noise. No clipping or reference fitting.')
