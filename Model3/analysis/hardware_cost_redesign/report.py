"""Render the bounded offline redesign decision from retained results."""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    folder=Path(__file__).parent
    search=json.loads((folder/'search_results.json').read_bytes())
    original=json.loads((folder/'verification.json').read_bytes())
    revised=json.loads((folder/'reallocated_budget/verification.json').read_bytes())
    ensembles=json.loads((folder/'ensemble_size_analysis.json').read_bytes())
    baseline=search['baseline']
    candidate=min([x for x in search['trials'] if x['steps']==48],key=lambda x:x['forward_maximum'])
    assert original['all_sampled_replays_passed'] and revised['candidate_ready_for_review']
    assert revised['search_sha256']==hashlib.sha256((folder/'search_results.json').read_bytes()).hexdigest()
    assert original['search_sha256']==revised['search_sha256']
    native0=original['runs']['baseline']['native_circuits'][0]
    native1=revised['runs']['candidate']['native_circuits'][0]
    assert native0['metrics']['two_qubit_gates']==1440 and native1['metrics']['two_qubit_gates']==960
    compact=json.loads((folder/'reallocated_budget/candidate_seed29_paths128.json').read_bytes())
    reference=json.loads((folder/'baseline_seed29_paths128.json').read_bytes())
    assert all(a==b[:48] for a,b in zip(compact['schedule']['jump_choices'],reference['schedule']['jump_choices']))
    physical=np.r_[.58,np.full(8,.5),.42]
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    colors=['#1565c0','#b34b12']
    for (label,row),color in zip([('Current cosine: 72 steps',baseline),('Warped cosine: 48 steps',candidate)],colors):
        C=np.asarray(row['output_covariance'])
        axes[0,0].plot(np.arange(1,row['steps']+1),row['step_dts'],color=color,label=label)
        axes[0,1].plot(np.arange(1,11),np.diag(C),'o-',color=color,label=label)
        axes[1,0].plot(np.arange(1,10),2*np.diag(C,1),'o-',color=color,label=label)
    axes[0,0].set(xlabel='Preparation step',ylabel='Trotter timestep h',title='Schedule values')
    axes[0,1].plot(np.arange(1,11),physical,'k--',label='Physical stationary populations')
    axes[0,1].set(xlabel='Site',ylabel='Population',title='All ten mean-channel output populations',xticks=range(1,11))
    axes[1,0].axhline(.32,color='black',ls='--',label='Physical stationary current / J')
    axes[1,0].set(xlabel='Bond',ylabel='Excitation current / J',title='All nine mean-channel output currents',xticks=range(1,10))
    axes[1,1].bar([0,1],[1440,960],color=colors)
    axes[1,1].set(xticks=[0,1],xticklabels=['Current 72','Candidate 48'],ylabel='Native CZ gates',title='Same chain, compilation settings and seed')
    for index,row in enumerate([native0,native1]):
        gates=row['metrics']['two_qubit_gates']
        axes[1,1].text(index,gates+30,f"{gates} CZ; depth {row['metrics']['depth']}\n{row['modeled_total_us']:.1f} us model",ha='center',fontsize=9)
    axes[1,1].set_ylim(0,1650)
    for ax in axes.flat:ax.grid(axis='y',alpha=.2)
    for ax in [axes[0,0],axes[0,1],axes[1,0]]:ax.legend(fontsize=7)
    fig.suptitle('Offline redesign: same Model3 stationary target and total tolerance 0.02\nMean/sampling allocation changes from 0.016/0.004 to 0.0165/0.0035; hardware noise is excluded',fontsize=11)
    for ext in ['png','pdf']:fig.savefig(folder/f'comparison.{ext}',dpi=170)
    plt.close(fig)
    trials=[]
    for row in search['trials']:
        trial={key:row.get(key) for key in ['family','steps','forward_maximum','output_error','estimated_cz','physical_time','evaluations','wall_seconds','polished']}
        trial['passed_original_mean_allowance_0_016']=row['passed']
        trials.append(trial)
    with (folder/'search_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=trials[0]);w.writeheader();w.writerows(trials)
    with (folder/'candidate_schedule.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['step','dt','physical_time_after_step'])
        for k,(dt,t) in enumerate(zip(candidate['step_dts'],candidate['time_nodes'][1:]),1):w.writerow([k,dt,t])
    with (folder/'ensemble_size_summary.csv').open('w',newline='') as f:
        flat=[{k:v for k,v in row.items() if k not in ['populations','currents']} for row in ensembles['rows']]
        w=csv.DictWriter(f,fieldnames=flat[0]);w.writeheader();w.writerows(flat)
    rows=['# Offline redesign decision: 48-step warped cosine', '',
        '**A cheaper candidate was found without changing Model3, its initial state, or the total 0.02 tolerance.** '
        'The 48-step schedule uses 960 CZ gates and 48 bath resets, one-third fewer than the current 72-step schedule. '
        'It requires a slightly different mean/sampling allocation and has passed three independent classical replay checks. '
        'No hardware job, live calibration query, noisy simulation or readout mitigation was performed.', '',
        '![Observable values, schedule and circuit costs](comparison.png)', '',
        '[PDF](comparison.pdf), [schedule CSV](candidate_schedule.csv), [search table](search_summary.csv), '
        '[48-step validation](reallocated_budget/verification.json), [baseline/64-step validation](verification.json).', '',
        '## Fixed question and decision', '',
        'Keep N=10, J=gamma=1, p_L=0.9, p_R=0.1, exchange prefactor 0.5, half-mixed preparation, '
        'Hamiltonian-first ordering, even-odd Hamiltonian splitting and rate-weighted sampling of one boundary branch per step. '
        'The physical stationary target remains n=(0.58,0.5,...,0.5,0.42), with excitation current I/J=0.32 on every bond. '
        'All ten populations and all nine currents enter the acceptance metric, even though hardware followup would measure populations first.', '',
        r'$$d(C,C_*)=\max\left\{\max_j|\Delta n_j|,\;\max_j|\Delta I_j|/J\right\}.$$', '',
        'The predeclared circuit-cost goal was at least 20% fewer CZ gates. With the original 0.016 mean allowance, '
        'the best passing tested schedule had 64 steps: only an 11.1% CZ reduction. The 48-step candidate narrowly failed '
        'that allowance. A separately declared reallocation to **0.0165 mean + 0.0035 sampling = 0.02 total** makes it feasible '
        'without loosening the total tolerance. It was then checked with a larger classical ensemble. This allocation change '
        'is part of the result, rather than a claim that the 48-step schedule passed the original 0.016 limit.', '',
        '## Schedule and acceptance', '',
        r'$$h_k=h_{\min}+\frac{h_{\max}-h_{\min}}{2}\left[1+\cos\left(\pi\left(\frac{k}{M-1}\right)^q\right)\right],\quad k=0,\ldots,M-1.$$', '',
        f"M=48, h_max={candidate['step_dts'][0]:.12f}, h_min={candidate['step_dts'][-1]:.12f}, q={candidate['parameters'][2]:.12f}. "
        f"The simulated preparation time is {candidate['physical_time']:.9f}, compared with {baseline['physical_time']:.9f} for the baseline. "
        'The exponent below 1 front-loads the decrease: a few large initial steps are followed by earlier refinement. '
        'This is a fixed, reference-assisted optimized schedule, not a new online feedback controller.', '',
        '| Quantity | Current cosine | 48-step candidate |', '|---|---:|---:|',
        f"| Mean error at output | {baseline['output_error']:.9f} | {candidate['output_error']:.9f} |",
        f"| Worst mean error over the forward window | {baseline['forward_maximum']:.9f} | {candidate['forward_maximum']:.9f} |",
        f"| Mean allowance | 0.016 | 0.0165 |", '| Sampling allowance | 0.004 | 0.0035 |',
        f"| Stationary bias of final-step mean channel | {baseline['final_dt_stationary_bias']:.9f} | {candidate['final_dt_stationary_bias']:.9f} |", '',
        f"The candidate was directly checked at every node for {candidate['validation_steps']} constant-final-step continuation steps, "
        f"from time {candidate['physical_time']:.6f} to {candidate['forward_times'][-1]:.6f}. This covers at least 10/J of future physical time. "
        'Those continuation steps are classical validation work and do not enter the preparation circuit. '
        'This matches the existing finite-window protocol; it is not a permanent-settling proof. '
        'Large early steps prepare a target state efficiently but do not imply accurate intermediate Lindblad dynamics.', '',
        '## Independent sampled checks', '',
        'After schedule selection and the new budget split, the candidate was replayed with **R=16,384** at seeds 29, 30 and 31. '
        'Each check requires the observed sampled output distance below 0.02, sampling distance below 0.0035 and maximum descriptive two-SE below 0.0035.', '',
        '| Seed | Sampled output error | Sampling distance | Maximum two-SE | Status |', '|---|---:|---:|---:|---|']
    for seed,row in revised['runs']['candidate']['replays'].items():
        a=row['assessment'];rows.append(f"| {seed} | {a['output_sampled_error']:.9f} | {a['max_sampling_distance']:.9f} | {a['max_two_standard_errors']:.9f} | {row['status']} |")
    rows += ['', 'The baseline and 64-step schedule also passed independent R=8192 checks at these three seeds. '
        'Trajectory means analytically average the mixed initial state; these checks exclude preparation-bit sampling, final measurement shots and hardware noise. '
        'Two-SE values and three seeds are descriptive checks, not simultaneous confidence guarantees. '
        'The larger classical R does not increase per-circuit gate depth and is not a requirement to execute 16,384 hardware circuits.', '',
        '## Native circuit cost and hardware timing', '',
        '| Quantity | Current 72-step, path 0 | Candidate 48-step, path 0 | Reduction |', '|---|---:|---:|---:|',
        '| CZ gates | 1440 | 960 | 33.33% |', '| Bath resets | 72 | 48 | 33.33% |', '| Two-qubit depth | 288 | 192 | 33.33% |',
        f"| Full native depth | {native0['metrics']['depth']} | {native1['metrics']['depth']} | {100*(1-native1['metrics']['depth']/native0['metrics']['depth']):.2f}% |",
        f"| Summed modeled block duration, including measurement | {native0['modeled_total_us']:.3f} us | {native1['modeled_total_us']:.3f} us | {100*(1-native1['modeled_total_us']/native0['modeled_total_us']):.2f}% |", '',
        'Compilation is entirely offline, on the same saved chain [39,49,48,58,57,56,66,76,86,85,95,105], '
        'optimization level 3 and transpiler seed 11. The native target was reconstructed from the saved 2026-10-25 calibration, '
        'using dt=4 ns, saved X/SX/CZ/measurement/reset durations and virtual RZ. It is not a live IBM target or a bit-for-bit '
        'reproduction of all backend metadata. Absolute compiled depths may differ slightly on a fresh target.', '',
        'The representative path-0 comparison uses identical branch choices for the first 48 steps. The baseline was also compiled '
        'for paths 1–3: depth 1126–1128 and summed modeled durations 113.708–128.340 us. '
        'The 48-step timing figure is one representative circuit, not a measured duration distribution. Final readouts can start '
        'at different modeled times on different sites; the summary includes the last final readout. '
        'Unknown dynamic reset-block latency is excluded. Actual hardware elapsed time, error reduction and 0.02 accuracy are not predicted by these percentages.', '',
        'The survival measurements motivate reducing duration, while the CZ/reset reductions make this a material implementation change. '
        'They do not provide a calibrated noise model from which to forecast the new observable values. No such forecast is claimed.', '',
        '## Finite-ensemble issue in the existing hardware protocol', '',
        'The first 128 seed-29 candidate paths have ideal population distance **0.030124**, already above 0.02. '
        'This is finite-path sampling, before any hardware noise, and prevents using that subset as evidence of 0.02 steady-state accuracy. '
        'The compact 128-path input is therefore an inspection artifact, not an independently accepted hardware ensemble.', '',
        'A separate fixed ensemble-size check, with no schedule or seed reselection, gives:', '',
        '| Paths | Baseline population-error range, seeds 29–31 | Candidate population-error range, seeds 29–31 |', '|---|---:|---:|']
    for R in [128,512,1024,4096]:
        ranges=[]
        for label in ['baseline','candidate48']:
            values=[r['population_error'] for r in ensembles['rows'] if r['method']==label and r['trajectories']==R]
            ranges.append(f"{min(values):.6f}–{max(values):.6f}")
        rows.append(f"| {R} | {ranges[0]} | {ranges[1]} |")
    rows += ['', 'For a single **population** comparison at the existing 8192-shot budget per schedule, '
        '**1024 trajectories × 8 shots** is a reasonable concrete allocation for both schedules: '
        'all three tested ideal population means are below 0.02. This does not mean the 1024-path ensembles satisfy the tighter '
        '0.0035 covariance-sampling allocation, nor does it establish a confidence bound or a current measurement claim. '
        'One 1024-path candidate current estimate still exceeds 0.02; the hardware followup is explicitly about populations. '
        'Measurement-shot uncertainty and device errors remain. Use the same fixed seed, chain, total shots and interleaved schedule order, '
        'and compare each hardware population vector with both its own finite-ensemble ideal and the physical stationary target.', '',
        '## Bounded search and stopping decision', '',
        f"The search tested cosine, warped cosine and two-stage families at 40, 48, 56 and 64 steps, with positive steps between 0.001 and 4. "
        f"It used 50 differential-evolution iterations per trial, followed by four local starts capped at 350 evaluations each. "
        f"The complete run used {sum(r['evaluations'] for r in search['trials'])} proxy evaluations. "
        'Every retained trial was independently checked by direct continuation at every validation node. '
        'The best 40-step result failed even the revised 0.0165 allowance. The search is heuristic and does not prove that 48 is minimal.', '',
        '**Retain the 48-step candidate for one coordinated comparison against the current 72-step baseline. Stop the diagnostic expansion here.** '
        'This is a concrete circuit-cost improvement with unchanged total classical tolerance; hardware improvement remains an experimental question. '
        'If that paired comparison does not show useful improvement, report that outcome rather than automatically launching another small control campaign.', '',
        '## Reproduction and storage', '',
        'Run from the repository root (Bash):', '', '```bash',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python Model3/analysis/hardware_cost_redesign/search.py',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python Model3/analysis/hardware_cost_redesign/polish.py',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python Model3/analysis/hardware_cost_redesign/verify.py',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python Model3/analysis/hardware_cost_redesign/verify.py --candidate-steps 48 --mean-tolerance .0165 --sampling-tolerance .0035 --trajectories 16384 --labels candidate --compile-count 1 --output Model3/analysis/hardware_cost_redesign/reallocated_budget/verification.json',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python Model3/analysis/hardware_cost_redesign/ensemble.py',
        'MPLCONFIGDIR=/tmp/model3_redesign_mpl .venv/bin/python Model3/analysis/hardware_cost_redesign/report.py',
        '```', '',
        'The full R=8192/16384 covariance dumps were computed in memory and not retained as bulky JSON files. '
        'Their canonical hashes, seeds, assessments and output observables are retained. Compact 128-path inputs preserve exact '
        'trajectory choices and per-path ideal covariances for circuit inspection; their provenance explicitly limits sampling validation '
        'to the original full replay. Neither hardware runner nor IBM account creation is invoked by these scripts.', '',
        '[Additional numerical checks](checks.json) compare the averaged channel with an explicit weighted sum over all four '
        'sampled alternatives at every candidate preparation/validation node, check covariance eigenvalues and symmetry, '
        'validate the three compact inputs and record the installed numerical package versions. These checks concern '
        'the implemented covariance channel; no new density-matrix or hardware noise simulation is implied.', '']
    (folder/'Report.md').write_text('\n'.join(rows),encoding='utf-8')
    print(folder/'Report.md')


if __name__=='__main__':main()
