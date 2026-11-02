"""Report actual Aer evidence separately from fixed-path shot-budget bounds."""
from pathlib import Path
import json
import sys

import numpy as np
from qiskit_aer import AerSimulator

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from library.reservoir_scaling import atomic_json
from Model2.Experiment5 import input_data,run_hardware,aer_sampling

DIRECTORY=Path(__file__).resolve().parent


def control_check():
    source=input_data.make_source(input_data.load_spec(),2,29)
    circuits=[];labels=[]
    for job in range(7):
        for name,population in [("zero",0.),("one",1.),("mixture",.5)]:
            body=run_hardware.build_body(source,0,steps=0,populations=[population]*10)
            circuits.append(run_hardware.circuits.append_readout(body,list(range(10)),"z"))
            labels.append((job,name,population))
    result=AerSimulator(method="statevector",max_parallel_threads=2,max_parallel_experiments=1).run(
        circuits,shots=2048,memory=True,seed_simulator=2029).result()
    if not result.success:
        raise RuntimeError(str(result))
    rows=[]
    for index,(job,name,target) in enumerate(labels):
        memory=aer_sampling.packed_memory(result.get_memory(index),circuits[index],10)
        populations=(((memory[:,None]>>np.arange(10))&1)).mean(0)
        rows.append({"job":job,"control":name,"maximum_error":float(max(abs(populations-target))),
                     "populations":populations.tolist(),"metadata":result.results[index].metadata})
    return {"shots_per_control":2048,"circuits":21,"rows":rows,
            "nominal_mixture_two_SE":float(2*np.sqrt(.25/2048)),
            "scope":"Actual noiseless Aer controls for seven pilot jobs; no device SPAM or simultaneous control-acceptance claim."}


def main():
    location=DIRECTORY/"runs/aer_sampling"
    studies=[(path,json.loads(path.read_text())) for path in location.glob("aer_*/study.json")]
    pilots=[(p,s) for p,s in studies if s["configuration"]["paths"]==1024 and s["configuration"]["shots"]==8
            and s["configuration"]["aer_seed"]==2026 and not s["configuration"]["full"] and s.get("complete")]
    references=[(p,s) for p,s in studies if s["configuration"]["full"] and s.get("complete")]
    if len(pilots)!=1 or not references:
        raise ValueError("complete the prescribed Aer pilot and full-bath reference first")
    pilot_path,pilot=pilots[0]
    ref_path,reference=references[0]
    exact=json.loads((location/"exact_reference.json").read_text())
    if not exact.get("passed"):
        raise ValueError("exact density reference must pass")
    budget_file=location/"shot_budget.json"
    if budget_file.exists():
        budget=json.loads(budget_file.read_text())
    else:
        source=input_data.make_source(input_data.load_spec(),16384,29)
        rows=aer_sampling.planning_rows(source,[1024,4096,8192,16384],[8,16,24,32,64,128])
        budget={"spec_sha256":input_data.SPEC_SHA256,"seed":29,"confidence":.95,"rows":rows,
                "recommended":next(r for r in rows if r["trajectories"]==16384 and r["shots"]==32)}
        atomic_json(budget_file,budget)
    controls_file=location/"controls.json"
    controls=json.loads(controls_file.read_text()) if controls_file.exists() else control_check()
    atomic_json(controls_file,controls)
    summary={"actual_Aer_pilot":{key:pilot[key] for key in ["configuration","maximum_target_error","maximum_mean_channel_error",
        "maximum_same_path_error","maximum_two_SE","empirical_checks_passed","total_simulation_seconds","versions"]},
        "pilot_source":str(pilot_path.relative_to(DIRECTORY)),"full_bath_source":str(ref_path.relative_to(DIRECTORY)),
        "exact_density_maximum_discrepancy":max(row["maximum_discrepancy"] for row in exact["paths"]),
        "exact_density_trace_range":[min(row["trace"] for row in exact["paths"]),max(row["trace"] for row in exact["paths"])],
        "shot_budget":budget,"Aer_controls_maximum_error":max(row["maximum_error"] for row in controls["rows"]),
        "scope":"Actual pilot/reference Aer simulations; larger-run sufficiency is an ideal conditional concentration calculation, not a completed full large Aer replay or hardware-noise guarantee."}
    atomic_json(DIRECTORY/"aer_validation_summary.json",summary)
    recommend=budget["recommended"]
    lines=["# Aer validation and trajectory/shot sufficiency","",
        "**The 1024-trajectory, 8-shot hardware pilot is insufficient for the original 0.02 accuracy / 0.004 sampling criteria.** Aer confirms substantial preparation/reset/final-measurement fluctuations in addition to the known finite-path discrepancy. It remains a device diagnostic, not a precision configuration.","",
        "All calculations preserve the exact 44-step schedule, N=10, J=1, Gamma=2, p=.25, half-mixture, jump-first even/odd order and fixed path seed 29. No schedule, seed or paths were reselected using the Aer outcome. No IBM account, live calibration or hardware job was used.","",
        "## Actual Aer evidence","",
        "The complete pilot simulated 1024 paths x 8 shots x all three bases: **24576 actual evolution-shot outcomes**, at simulator seed 2026. Freshly reset baths were traced into exact one-site Kraus maps, retaining the same system channel and shot distribution. Initial ignored measurements become their nonselective dephasing channel, which Aer statevector unravels stochastically. Counts come from final measurements; no expectation snapshots replace measurement shots.","",
        "| Pilot quantity | Result |","|---|---:|",
        f"| Maximum Aer error to physical stationary target | {pilot['maximum_target_error']:.9g} |",
        f"| Maximum Aer error to infinite-path mean channel | {pilot['maximum_mean_channel_error']:.9g} |",
        f"| Maximum Aer error to its own finite-path ideal | {pilot['maximum_same_path_error']:.9g} |",
        f"| Maximum descriptive across-path two SE, including shots | {pilot['maximum_two_SE']:.9g} |",
        f"| Original empirical checks passed | {pilot['empirical_checks_passed']} |", "",
        "![Aer pilot profiles and matched references](aer_pilot.png)","",
        "The exact finite-path ideal already has target error 0.0201363873 and mean-channel distance 0.00680954294, above the respective allowances before shot noise. Increasing shots with these same 1024 paths cannot repair that expected finite-path profile.","",
        "Two independent reference checks support the bath reduction and circuit implementation:","",
        f"- Exact ten-qubit Aer density propagation at paths 0 and 1 agrees with the covariance predictions for every population, current and both individual XY/YX correlators to **{summary['exact_density_maximum_discrepancy']:.3g}**. Density traces are within 3e-14 of one.",
        f"- Full **twelve-qubit** circuits at paths 0 and 1, 128 shots per basis, seed 2027, retain both actual bath resets and preparation records: **768 actual outcomes**. Their maximum observed same-path discrepancy is {reference['maximum_same_path_error']:.6g}; this low-path reference is not an accuracy-validated ensemble. Small-system exact density tests separately prove equality of original, lowered and bath-traced circuits for all four jump types at the actual large angles.","",
        f"Actual Aer controls also ran all 21 short preparation circuits for seven pilot jobs at 2048 shots each. Largest local discrepancy: **{summary['Aer_controls_maximum_error']:.6g}**. For a half-mixture, nominal pointwise two SE is {controls['nominal_mixture_two_SE']:.6g}; these controls are diagnostics, not a simultaneous 0.02 calibration certificate.","",
        "## Larger-run recommendation: calculated, not a completed full Aer replay","",
        "Use **16384 trajectories x 32 shots per basis, fixed path seed 29**, when assessing the original precision criteria in the ideal model. The entire 1.57-million-shot configuration was **not** replayed in Aer. Its statistical sufficiency is established using the exact finite-path probabilities, validated channel construction, and the conditional bound below. The Aer pilot and the analytic recommendation are distinct evidence.","",
        "| Paths | Shots per basis | Conditional failure-probability upper bound | At least 95% ideal-shot coverage? | Total hardware shots incl. controls |",
        "|---|---:|---:|---|---:|"]
    for row in budget["rows"]:
        if (row["trajectories"],row["shots"]) in [(1024,8),(4096,128),(8192,64),(8192,128),(16384,8),(16384,16),(16384,24),(16384,32),(16384,64)]:
            lines.append(f"| {row['trajectories']} | {row['shots']} | {row['simultaneous_failure_probability_bound']:.9g} | {row['sufficient']} | {row['hardware_shots_including_controls']} |")
    lines += ["", "A bound above 0.05 means this calculation does not certify that configuration; it is not an exact failure probability. The 24-shot option also meets the 95% condition, while 32 shots gives more margin. The recommended configuration has:","",
        f"- Ideal same-path target error **{recommend['ideal_target_error']:.9g}** and mean-channel distance **{recommend['ideal_sampling_distance']:.9g}**.",
        f"- Probability at most **{recommend['simultaneous_failure_probability_bound']:.9g}** of violating either local-observable requirement, jointly over all ten populations and nine currents, conditional on this exact finite ensemble, independent ideal shots and the verified channel.",
        f"- **{recommend['jobs']} jobs**, **{recommend['evolution_shots']} evolution shots**, **{recommend['hardware_shots_including_controls']} shots including the existing three 2048-shot controls per job** at batch size 512.","",
        "```powershell",
        "python Model2/Experiment5/run_hardware.py --execute --backend ibm_phoenix --trajectory-count 16384 --seed 29 --shots 32 --bases z xy yx --batch-size 512 --control-shots 2048",
        "```","",
        "The pilot defaults remain unchanged to avoid silently expanding a run from 67584 to over two million hardware shots. This command is explicit about that larger workload. The bound excludes hardware bias, drift, preparation/readout defects and correlated device shots. More shots cannot remove those effects. No hardware accuracy guarantee follows from noiseless Aer.","",
        "## Conditional probability calculation","",
        "For fixed path r, the final Z bit is Bernoulli with its exact covariance-predicted population n_r. The current estimator combines two independent Pauli measurements. Number conservation gives <YX>=-<XY>=2 C_(j,j+1) in the stored real gauge; the density checks verify both correlators individually. Thus the conditional variances of the averaged observables are",
        "",
        "```text",
        "V_population = mean_r[n_r(1-n_r)] / (R*S)",
        "V_current/J = 2*c^2 * mean_r[1-4*C_r,j,j+1^2] / (R*S),  c=1/4.",
        "```","",
        "A centered shot contribution is bounded by b=1/(R*S) for population and b=2*c/(R*S) for current/J. Bernstein's inequality bounds each one-sided tail by exp(-d^2/[2*(V+b*d/3)]); see Theorem 2.9.5 in [Vershynin's author manuscript](https://www.math.uci.edu/~rvershyn/papers/HDP-book/HDP-2.pdf). The application and numerical thresholds here are our calculation.","",
        "For each observable, intersect the allowed physical-target interval ±0.02 with its infinite-path-mean interval ±0.004. Compute the upper and lower distances from that observable's exact finite-path expectation to this intersection, then sum all 38 one-sided bounds. A nonpositive distance gives an uncertified bound of one. This union bound allows arbitrary correlations between observables measured in the same shot; independence is required across ideal shots and the separately measured bases. Conditional shot variance already includes preparation and reservoir-reset randomness; it is not added again.","",
        "The saved budget JSON also contains a simpler uniform-confidence-width diagnostic. For 16384x32 its scalar sampling upper bound is 0.004018, slightly above 0.004. That shortcut is more conservative: the passing decision uses the summed actual per-observable tail bounds, not the worst uniform-width scalar. The latter does not invalidate the sharper joint bound. These confidence statements apply to the prescribed fixed ensemble and configuration, not jointly to every configuration in the table.","",
        "## Reproduction and artifacts","",
        "```bash",
        "python Model2/Experiment5/aer_sampling.py --resume",
        "python Model2/Experiment5/aer_sampling.py --paths 2 --shots 128 --full --aer-seed 2027 --resume",
        "python Model2/Experiment5/aer_reference.py",
        "python -m Model2.Experiment5.summarize_aer",
        "```","",
        "Aer runs checkpoint every 16 paths. NPZ files store actual packed final-register outcomes; metadata includes sample hashes, simulator settings and per-chunk times. Full and reduced circuits preserve the initial mixture and intended bath channel. Reformatting preparation bits cannot change final-bit decoding. Tests cover exact channel equivalence, final-register offsets, both current parities, insufficient fixed ensembles, confidence/workload calculations and verified resume.","",
        f"Actual pilot: `{summary['pilot_source']}`. Full-bath reference: `{summary['full_bath_source']}`. Exact density and shot-budget files are under `runs/aer_sampling/`; raw generated data stay local. A compact [summary](aer_validation_summary.json) is retained. Simulator version: **Aer {pilot['versions']['qiskit-aer']}**."]
    (DIRECTORY/"AerValidation.md").write_text("\n".join(lines)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(10,7),constrained_layout=True)
    for ax,segment,label in [(axes[0],slice(0,10),"Population"),(axes[1],slice(10,19),"Current / J")]:
        values=np.asarray(pilot["observables"])[segment]
        x=np.arange(1,len(values)+1)
        ax.errorbar(x,values,yerr=2*np.asarray(pilot["across_path_standard_errors"])[segment],fmt="o",label="Aer pilot ±2 across-path SE")
        ax.plot(x,np.asarray(pilot["same_path_ideal"])[segment],"--",label="Exact same-path ideal")
        ax.plot(x,np.asarray(pilot["mean_channel"])[segment],"-.",label="Infinite-path mean")
        ax.plot(x,np.asarray(pilot["physical_target"])[segment],":",color="black",label="Physical target")
        ax.set(xlabel="Site / bond (one based)",ylabel=label)
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    for suffix in ("png","pdf"):
        fig.savefig(DIRECTORY/f"aer_pilot.{suffix}",dpi=160)
    plt.close(fig)
    print(f"Saved {DIRECTORY/'AerValidation.md'}",flush=True)
    return summary


if __name__=="__main__":
    main()
