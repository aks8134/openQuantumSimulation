"""Audit completed full-study checkpoints and produce standalone scaling figures."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from FinalModel.Experiment1 import full_study,study
from library import reservoir_scaling as physics


def fit_power(rows,key,minimum=20):
    group=[r for r in rows if r["N"]>=minimum and r[key]>0]
    if len(group)<3:return None
    x=np.log([r["N"] for r in group]);y=np.log([r[key] for r in group])
    slope,intercept=np.polyfit(x,y,1)
    residual=y-(slope*x+intercept)
    denominator=float(np.sum((y-y.mean())**2))
    return dict(exponent=float(slope),coefficient=float(np.exp(intercept)),
                log_R_squared=None if denominator<1e-20 else 1-float(np.sum(residual**2))/denominator,
                N_range=[group[0]["N"],group[-1]["N"]],points=len(group),
                interpretation="empirical fit over computed sizes, not an asymptotic claim")


def collect(directory):
    provenance=json.loads((directory/"configuration.json").read_text())
    config=provenance["configuration"]
    if provenance!=full_study.provenance(config):raise ValueError("current source or package versions differ from run provenance")
    rows=[]
    audit=dict(exact_cases=0,validated_schedules=0,compiled_paths=0,maximum_kernel_disagreement=0.,
               maximum_stationary_residual=0.,failures=[])
    audit["numerical_kernel_compiler"]=subprocess.run(["g++","--version"],check=True,capture_output=True,text=True).stdout.splitlines()[0]
    audit["python_version"]=sys.version
    audit["analysis_source_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for name,spec in config["models"].items():
        if [spec["J"],spec["gamma"],spec["prefactor"]]!=[1.,1.,.5]:raise ValueError("normalization differs from agreed study")
        for n in config["sizes"]:
            exact=json.loads((directory/f"exact_{name}_N{n}.json").read_text())["result"]
            schedules=json.loads((directory/f"schedules_{name}_N{n}.json").read_text())["result"]
            resources=json.loads((directory/f"resources_{name}_N{n}.json").read_text())["result"]
            if exact["status"]!="permanent_bound_met":raise ValueError("incomplete continuous reference")
            audit["exact_cases"]+=1
            audit["maximum_stationary_residual"]=max(audit["maximum_stationary_residual"],exact["stationary_residual"])
            row=dict(model=name,N=n,settling_time_upper_bound=exact["settling_time_upper_bound"])
            for family,selection in schedules["families"].items():
                if selection["status"]!="mean_forward_window_passed":raise ValueError(f"No accepted schedule: {name} N={n} {family}")
                K=selection["selected_steps"]
                dts=np.asarray(selection["step_dts"])
                validation=selection["validation"]
                if K!=len(dts) or abs(sum(dts)-selection["physical_time"])>1e-7:raise ValueError("schedule accounting mismatch")
                if not validation["passed"] or max(validation["forward_errors"])>config["mean_tolerance"]:raise ValueError("acceptance mismatch")
                if len(validation["forward_errors"])!=validation["forward_steps"]+1:raise ValueError("missing forward nodes")
                if validation["forward_duration"]<config["forward_window"]-1e-10:raise ValueError("short forward window")
                covariance=np.asarray(validation["output_covariance"])
                if np.linalg.norm(covariance-covariance.T)>1e-9:raise ValueError("covariance is not symmetric")
                values=np.linalg.eigvalsh(covariance)
                if values[0]<-1e-9 or values[-1]>1+1e-9:raise ValueError("unphysical covariance")
                audit["validated_schedules"]+=1
                audit["maximum_kernel_disagreement"]=max(audit["maximum_kernel_disagreement"],selection.get("kernel_maximum_disagreement",0.))
                selected=[r for r in resources["records"] if r["family"]==family]
                if len(selected)!=config["resource_paths"] or any(r["status"]!="compiled" for r in selected):raise ValueError("incomplete resource paths")
                for resource in selected:
                    if resource["steps"]!=K or resource["logical_exchanges"]!=n*K:raise ValueError("resource schedule mismatch")
                    if resource["native_operations"]["reset"]!=K:raise ValueError("reset accounting mismatch")
                    if resource["logical_qubits"]!=n+2:raise ValueError("ancilla accounting mismatch")
                    audit["compiled_paths"]+=1
                row.update({f"{family}_steps":K,f"{family}_time":selection["physical_time"],
                    f"{family}_error":validation["maximum_forward_error"],
                    f"{family}_native_cx":float(np.mean([r["native_operations"].get("cx",0) for r in selected])),
                    f"{family}_native_depth":float(np.mean([r["native_depth"] for r in selected])),
                    f"{family}_native_depth_min":min(r["native_depth"] for r in selected),
                    f"{family}_native_depth_max":max(r["native_depth"] for r in selected),
                    f"{family}_T_estimate":float(np.mean([r["fault_tolerant"]["T_count_leading_order_estimate"] for r in selected])),
                    f"{family}_base_Clifford":float(np.mean([r["fault_tolerant"]["Clifford_gates_excluding_RZ"] for r in selected]))})
                if family=="warped_cosine":
                    row["warped_output_total_excitation"]=validation["output_total_excitation"]
                    row["warped_nonconstant"]=bool(np.ptp(dts)>1e-10)
            row["warped_reduction_vs_fixed"]=1-row["warped_cosine_steps"]/row["fixed_steps"]
            row["warped_reduction_vs_constant"]=1-row["warped_cosine_steps"]/row["constant_steps"]
            rows.append(row)
    return config,rows,audit


def plots(directory,rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    families=[("fixed","Fixed dt=0.05"),("constant","Searched constant"),("warped_cosine","Warped cosine family")]
    models=sorted({r["model"] for r in rows})
    def save(fig,name):
        fig.tight_layout();fig.savefig(directory/f"{name}.png",dpi=200);fig.savefig(directory/f"{name}.pdf");plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13.5,4))
    for ax,name in zip(axes,models):
        group=[r for r in rows if r["model"]==name]
        for family,label in families:ax.loglog([r["N"] for r in group],[r[f"{family}_steps"] for r in group],"o-",label=label)
        ax.set(title=name,xlabel="System qubits N",ylabel="Executable preparation steps K");ax.grid(alpha=.2,which="both")
    axes[0].legend(fontsize=8);save(fig,"steps_scaling")
    fig,axes=plt.subplots(2,3,figsize=(13.5,7.5))
    for col,name in enumerate(models):
        group=[r for r in rows if r["model"]==name]
        for row,metric,label in ((0,"native_cx","Compiled CX count"),(1,"native_depth","Compiled dependency depth")):
            ax=axes[row,col]
            for family,legend in families:ax.loglog([r["N"] for r in group],[r[f"{family}_{metric}"] for r in group],"o-",label=legend)
            ax.set(title=name,xlabel="System qubits N",ylabel=label);ax.grid(alpha=.2,which="both")
    axes[0,0].legend(fontsize=8);save(fig,"compiled_scaling")
    fig,axes=plt.subplots(2,3,figsize=(13.5,7.5))
    for col,name in enumerate(models):
        group=[r for r in rows if r["model"]==name]
        for row,metric,label in ((0,"T_estimate","Leading-order logical T estimate"),(1,"base_Clifford","Clifford gates before RZ synthesis")):
            ax=axes[row,col]
            for family,legend in families:ax.loglog([r["N"] for r in group],[r[f"{family}_{metric}"] for r in group],"o-",label=legend)
            ax.set(title=name,xlabel="System qubits N",ylabel=label);ax.grid(alpha=.2,which="both")
    axes[0,0].legend(fontsize=8);save(fig,"clifford_T_scaling")
    fig,axes=plt.subplots(3,3,figsize=(13.5,9))
    for i,name in enumerate(models):
        for j,n in enumerate((10,50,100)):
            data=json.loads((directory/f"schedules_{name}_N{n}.json").read_text())["result"]["families"]["warped_cosine"]
            axes[i,j].plot(np.arange(1,len(data["step_dts"])+1),data["step_dts"])
            axes[i,j].set(title=f"{name}, N={n}",xlabel="Preparation step",ylabel="Physical timestep")
            axes[i,j].grid(alpha=.2)
    save(fig,"selected_schedules")


def report(directory,config,rows,audit):
    with (directory/"scaling_summary.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fits={name:{key:fit_power([r for r in rows if r["model"]==name],key)
                for key in ("settling_time_upper_bound","fixed_steps","constant_steps","warped_cosine_steps","warped_cosine_native_cx","warped_cosine_T_estimate")}
          for name in config["models"]}
    physics.atomic_json(directory/"audit.json",audit)
    physics.atomic_json(directory/"fits.json",fits)
    lines=["# Full scaling study: J=gamma=1, H=(J/2) sum(XX+YY)","",
        f"Completed {len(rows)} model/size cases through N=100: {audit['validated_schedules']} independently validated mean schedules and {audit['compiled_paths']} compiled resource paths.","",
        "All models use J=gamma=1 and c=1/2. Reservoirs remain (0,0), (0.25,0.25), and (0.9,0.1). Model1 starts with a central excitation; Models2/3 start with a half mixture. These preparations remain different.","",
        "Times are physical evolution times in units of 1/J, not classical computation seconds. The continuous time is a sufficient permanent local-observable settling-time **upper bound**. Digital schedules pass every node of a finite continuation of at least 10 physical-time units. These are different guarantees. No many-body trace-distance mixing time, finite-trajectory accuracy, measurement-shot cost or hardware performance is claimed.","",
        "## N=100 results","",
        "| Model | Continuous time upper bound | Fixed steps | Constant steps | Warped-family steps | Warped physical time | Reduction vs fixed | Reduction vs constant |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r["N"]==100:lines.append(f"| {r['model']} | {r['settling_time_upper_bound']:.6g} | {r['fixed_steps']} | {r['constant_steps']} | {r['warped_cosine_steps']} | {r['warped_cosine_time']:.6g} | {100*r['warped_reduction_vs_fixed']:.2f}% | {100*r['warped_reduction_vs_constant']:.2f}% |")
    lines += ["","These are the best passing schedules found, not globally minimal counts. The warped family includes constant schedules and retains a constant incumbent if no shorter schedule is found.","",
        "![Continuous permanent local-settling time upper bounds](exact.png)","",
        "![Preparation-step scaling](steps_scaling.png)","",
        "## Compiled resources and logical Clifford+T estimates","",
        "The compiler transpiles symbolic two-qubit exchange blocks to the native RZ/SX/X/CX basis, then binds angles and composes blocks without inter-block optimization. Counts and dependency depths are computed exactly by streaming the compiled instructions' dependencies; no full-circuit size cap or large-N extrapolation is used. All interactions lie on the line with boundary-local ancillas, so line and all-to-all costs coincide for this compiler. Depth is not a pulse duration.","",
        "| Model at N=100 | Logical qubits | Warped CX | Warped native depth (path mean) | Base Clifford gates | Leading-order T estimate |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r["N"]==100:lines.append(f"| {r['model']} | 102 | {r['warped_cosine_native_cx']:.0f} | {r['warped_cosine_native_depth']:.1f} | {r['warped_cosine_base_Clifford']:.0f} | {r['warped_cosine_T_estimate']:.0f} |")
    lines += ["","Resources are per Z-readout path, averaged over two fixed random paths. This is a compilation sample, not a converged trajectory ensemble. Preparation measurements, K bath resets and terminal measurements are included.","",
        "Base Clifford counts exclude every RZ awaiting classification/synthesis. T estimates use exact T-angle detection plus the leading 3 log2(1/eta) model from [Ross and Selinger](https://arxiv.org/abs/1403.2975), with eta=0.001/[2 max(1,4c) R] for R generic rotations. Omitted terms mean these estimates are neither synthesized counts nor certified bounds. Exact post-synthesis Clifford counts, T-depth and physical error-correction costs are not computed.","",
        "![Compiled resources](compiled_scaling.png)","","![Logical Clifford and T resources](clifford_T_scaling.png)","",
        "## Full accepted step-count table","",
        "| Model | N | Continuous time upper bound | Fixed steps | Constant steps | Warped-family steps | Warped max forward error |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:lines.append(f"| {r['model']} | {r['N']} | {r['settling_time_upper_bound']:.6g} | {r['fixed_steps']} | {r['constant_steps']} | {r['warped_cosine_steps']} | {r['warped_cosine_error']:.8f} |")
    lines += ["","## Validation and interpretation","",
        f"All {audit['validated_schedules']} output covariances are physical in floating point, and every saved continuation node passes the 0.016 mean tolerance. Selection used a 0.5% numerical margin. The largest accelerated-versus-independent dense output discrepancy was {audit['maximum_kernel_disagreement']:.3g}; the largest stationary-equation residual was {audit['maximum_stationary_residual']:.3g}.","",
        "The 0.003 sampling allowance and 0.001 synthesis allowance remain separate from mean validation. No sampled-trajectory or hardware execution is included. Schedules optimize stationary-observable preparation without imposing a transient-matching tolerance. Large reductions versus dt=0.05 are not all attributable to cosine warping; the searched-constant comparison isolates the additional schedule-family benefit.","",
        "In Model1, small local errors can coexist with a nonzero total excitation. The CSV retains the output total excitation so a local-observable pass cannot be mistaken for full vacuum-state preparation.","",
        "| Model1 N | Total excitation at warped output |",
        "|---:|---:|"]
    for r in rows:
        if r["model"]=="model1" and r["N"] in (10,50,100):lines.append(f"| {r['N']} | {r['warped_output_total_excitation']:.8f} |")
    lines += ["","![Selected timestep schedules](selected_schedules.png)","",
        "## Empirical fits","",
        "Log-log fits use N>=20. They summarize this range and protocol; they do not establish asymptotic laws. The continuous certificate and digital finite window may produce different apparent exponents.","",
        "| Model | Quantity | Fitted exponent | Log-space R squared |",
        "|---|---|---:|---:|"]
    for name,mapping in fits.items():
        for key,fit in mapping.items():
            if fit:
                r_squared="undefined (constant values)" if fit["log_R_squared"] is None else f"{fit['log_R_squared']:.5f}"
                lines.append(f"| {name} | {key} | {fit['exponent']:.4f} | {r_squared} |")
    lines += ["","## Reproduce","","```bash",
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/finalmodel_mpl \\",
        "  .venv/bin/python FinalModel/Experiment1/full_study.py --workers 4",
        f"OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/finalmodel_mpl .venv/bin/python FinalModel/Experiment1/analyze_full.py {directory.relative_to(ROOT)}",
        "```","","Configuration, source/package provenance and full numerical checkpoints are saved locally. Compact tables, reports and PNG/PDF figures are portable. This report was generated from audited checkpoints.",""]
    (directory/"Results.md").write_text("\n".join(lines))
    relative=directory.relative_to(study.DIRECTORY)
    (study.DIRECTORY/"Results.md").write_text("# Full study results\n\nAll three models use J=gamma=1 and H=(J/2) sum(XX+YY).\n\n"
        f"The complete [results report]({relative}/Results.md) contains continuous references, fixed/constant/warped step counts, compiled resources and logical Clifford+T estimates through N=100.\n\n"
        f"Completed: {audit['exact_cases']} continuous references, {audit['validated_schedules']} independently validated mean schedules, and {audit['compiled_paths']} compiled path resource records.\n\n"
        "Continuous times are permanent local-settling upper bounds; digital acceptance is a finite forward-window check. T counts are leading-order estimates. No hardware execution was used.\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("directory",type=Path);args=parser.parse_args()
    directory=args.directory.resolve()
    config,rows,audit=collect(directory)
    plots(directory,rows);report(directory,config,rows,audit)
    print(json.dumps(audit,indent=2));print(directory/"Results.md")


if __name__=="__main__":main()
