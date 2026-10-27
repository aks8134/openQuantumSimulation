"""Review pulled mitigation counts offline, preserving the original run files.

Run from the repository root with:
    python -m Model3.analysis.offline_audit.analyze_mitigation_result
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.stats import norm

from Model3.Experiment5.analyze_hardware import serializable
from Model3.Experiment5.run_mitigation import analyze, save_analysis
from library.reservoir_scaling import atomic_json

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN = ROOT / 'Model3/Experiment5/hardware_runs/ibm_phoenix_mitigation_20261026T184843_413808Z'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-directory', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--bootstrap', type=int, default=10000)
    args = parser.parse_args()
    run = args.run_directory.resolve()
    output = ROOT / 'Model3/Experiment5/analysis/hardware' / run.name / 'mitigation_review'
    originals = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in run.iterdir() if p.is_file()}
    manifest = json.loads((run / 'manifest.json').read_bytes())
    cached = json.loads((run / 'hardware_results.json').read_bytes())
    with zipfile.ZipFile(run / 'results_bundle.zip') as archive:
        assert archive.testzip() is None
        for name in ('manifest.json', 'source.json', 'backend_snapshot.json', 'hardware_results.json'):
            assert json.loads(archive.read(name)) == json.loads((run / name).read_bytes())
    report = analyze(run, replicates=args.bootstrap, seed=29)
    for name, profile in report['statistics']['profiles'].items():
        np.testing.assert_allclose(profile['populations'], cached['statistics']['profiles'][name]['populations'], atol=1e-15, rtol=0)
    np.testing.assert_array_equal(report['per_path_populations'], cached['per_path_populations'])
    save_analysis(report, output)
    stats = report['statistics']
    profiles = stats['profiles']
    cfg = report['configuration']
    values = np.asarray(report['per_path_populations'])
    ideal_paths = np.asarray(report['per_path_ideal_populations'])
    ideal = np.asarray(report['ideal_same_paths'])
    target = np.asarray(report['physical_target'])
    weights = np.asarray(stats['linear_zne']['weights'])

    # Paired error comparisons without SPAM inversion. Reweight the ideal along
    # with each saved trajectory, and keep observed job sizes fixed.
    rng = np.random.default_rng(20261026)
    bootstrap_weights = np.zeros((args.bootstrap, len(values)))
    batch_rows = []
    for batch in manifest['batches']:
        records = manifest['circuits'][batch['start']:batch['stop']]
        indices = [r['trajectory'] for r in records if r['kind'] == 'trajectory' and r['arm'] == 'raw']
        size = len(indices)
        bootstrap_weights[:, indices] = rng.multinomial(size, np.full(size, 1 / size), size=args.bootstrap) / len(values)
        local_ideal = ideal_paths[indices].mean(axis=0)
        local_mean = values[indices].mean(axis=0)
        local_profiles = {'raw': local_mean[0], 'twirled': local_mean[1], 'zne': weights @ local_mean[1:]}
        batch_rows.append(dict(job_id=batch['job_id'], trajectories=size,
            running=batch['job_metrics']['timestamps']['running'],
            finished=batch['job_metrics']['timestamps']['finished'],
            controls=[r['case_id'] for r in records if r['kind'] == 'control'],
            populations=local_profiles, ideal_same_paths=local_ideal,
            ideal_mae={k: float(abs(v-local_ideal).mean()) for k, v in local_profiles.items()}))
    sampled = np.einsum('br,rkj->bkj', bootstrap_weights, values)
    sampled_ideal = bootstrap_weights @ ideal_paths
    sampled_profiles = {'raw': sampled[:, 0], 'twirled': sampled[:, 1],
                        'zne': np.einsum('k,bkj->bj', weights, sampled[:, 1:])}
    comparisons = []
    for method, reference in [('twirled', 'raw'), ('zne', 'raw'), ('zne', 'twirled')]:
        for metric, reduce in [('mae', np.mean), ('maximum', np.max)]:
            differences = (reduce(abs(sampled_profiles[method]-sampled_ideal), axis=1)
                           - reduce(abs(sampled_profiles[reference]-sampled_ideal), axis=1))
            comparisons.append(dict(method=method, reference=reference, metric=metric,
                difference=profiles[method]['ideal_'+metric]-profiles[reference]['ideal_'+metric],
                ci95=np.quantile(differences, [.025, .975])))

    # Does the held-out mixture agree with the local affine calibration? These
    # controls occur in different jobs, so a failure cannot identify its cause.
    controls = []
    for arm, calibration in stats['calibration'].items():
        a, b, mix = [np.asarray(calibration[k]['populations']) for k in ('all_zero', 'all_one', 'initial_mixture')]
        variances = [np.diag(calibration[k]['covariance']) for k in ('all_zero', 'all_one', 'initial_mixture')]
        predicted = (a+b)/2
        se = np.sqrt(variances[2]+(variances[0]+variances[1])/4)
        for j in range(len(ideal)):
            controls.append(dict(arm=arm, site=j+1, physical_qubit=cfg['physical_path'][j+1],
                zero=a[j], one=b[j], mixture=mix[j], predicted_mixture=predicted[j],
                corrected_mixture=stats['heldout_mixture'][arm]['corrected'][j],
                corrected_ci95=np.asarray(stats['heldout_mixture'][arm]['bootstrap_ci95'])[:, j],
                residual_se=se[j], approximate_two_sided_p=2*norm.sf(abs(mix[j]-predicted[j])/se[j])))
    order = np.argsort([c['approximate_two_sided_p'] for c in controls])
    adjusted = np.minimum(1, np.maximum.accumulate(np.array([controls[i]['approximate_two_sided_p'] for i in order]) * (len(order)-np.arange(len(order)))))
    for index, p in zip(order, adjusted):
        controls[index]['holm_p_20_controls_normal_approximation'] = p

    metrics = []
    for arm in ('raw', 'twirled'):
        for factor in ([1.] if arm == 'raw' else cfg['fold_factors']):
            records = [r for r in manifest['circuits'] if r['kind'] == 'trajectory' and r['arm'] == arm and r['requested_factor'] == factor]
            metrics.append(dict(arm=arm, factor=factor, circuits=len(records),
                cz_counts=sorted({r['metrics']['operation_counts']['cz'] for r in records}),
                reset_counts=sorted({r['metrics']['operation_counts']['reset'] for r in records}),
                depth_min=min(r['metrics']['depth'] for r in records), depth_max=max(r['metrics']['depth'] for r in records)))
    review = dict(reconstruction_matches_saved_populations=True, bundle_matches_local_files=True,
        verification=report['verification'], comparisons_without_spam=comparisons,
        heldout_controls=controls, jobs=batch_rows, submitted_metrics=metrics,
        raw_file_sha256=originals,
        scope='Observed jobs and randomizations only; bootstrap intervals are descriptive, not simultaneous. No drift or calibration-model uncertainty is included.')
    atomic_json(output / 'review.json', serializable(review))
    rows = []
    for j in range(len(ideal)):
        row = dict(site=j+1, physical_qubit=cfg['physical_path'][j+1], physical_target=target[j], ideal_same_paths=ideal[j])
        row.update({name: p['populations'][j] for name, p in profiles.items()})
        rows.append(row)
    with (output / 'all_site_populations.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    plot_populations(output, report)

    lines = ['# Review of the pulled mitigation pilot', '',
        'Both jobs completed: 847 circuits, 77,824 shots, 64 matched branch trajectories, and 256 shots per trajectory per arm/factor. '
        'The first-order 48-step schedule and 48 bath resets are retained. Raw counts reconstruct every saved population exactly; '
        'the transfer archive matches its local raw files and source hash checks pass. Native QPY files were not transferred, so this review '
        'cannot independently recheck compiled circuit contents. The execution-machine cached report records both QPY batches as verified.', '',
        '![Three preset observable populations with uncertainty](selected_populations.png)', '',
        'The error bars above are 1.96 times conditional shot/calibration SE for the fixed saved branch ensemble. '
        'The separate populations.csv contains descriptive trajectory-bootstrap intervals. Neither includes hardware drift or calibration-model bias.', '',
        '| Method | Same-path ideal MAE | Same-path ideal maximum | Physical-target maximum |',
        '|---|---:|---:|---:|']
    for name, p in profiles.items():
        lines.append(f"| {name} | {p['ideal_mae']:.6f} | {p['ideal_maximum']:.6f} | {p['physical_maximum']:.6f} |")
    lines += ['', '## What the mitigation changed', '',
        'Twirling reduces the largest same-path discrepancy substantially, mainly by moving site 9 toward its ideal. '
        'Its average error changes little, and several sites worsen. CZ-only linear ZNE improves the point estimates further, '
        'but the intervals for its incremental benefit over twirling include no improvement. '
        'Twirling here combines CZ Pauli frames and terminal measurement flips, so their individual contributions are not isolated.', '',
        '| Comparison (negative favors method) | Metric | Difference | Descriptive paired 95% interval |',
        '|---|---|---:|---|']
    for c in comparisons:
        lo, hi = c['ci95']
        lines.append(f"| {c['method']} minus {c['reference']} | {c['metric']} | {c['difference']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines += ['', '## Observable numbers', '',
        '| Site | Physical qubit | Physical stationary | Ideal: same 64 paths | Raw | Twirled | ZNE |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['site']} | {row['physical_qubit']} | {row['physical_target']:.6f} | {row['ideal_same_paths']:.6f} | {row['raw']:.6f} | {row['twirled']:.6f} | {row['zne']:.6f} |")
    lines += ['', '## Preparation-control failure', '',
        'Site 3 is physical qubit 48. Its mixture is inconsistent with the zero/one affine prediction under both readout treatments:', '',
        '| Treatment | Measured mixture | Predicted from zero/one | Corrected mixture | Bootstrap 95% interval |',
        '|---|---:|---:|---:|---|']
    for c in controls:
        if c['site'] == 3:
            lo, hi = c['corrected_ci95']
            lines.append(f"| {c['arm']} | {c['mixture']:.6f} | {c['predicted_mixture']:.6f} | {c['corrected_mixture']:.6f} | [{lo:.6f}, {hi:.6f}] |")
    lines += ['',
        'The expected corrected mixture is 0.5. Both discrepancies also survive a Holm adjustment across 20 control checks '
        'using approximate normal tests (details in review.json). This flags a failed control assumption, not a uniquely identified '
        'bad-qubit mechanism: preparation, measurement context, and drift remain possible. Zero/one controls were in job 1; '
        'the mixture controls were in job 2. The jobs began roughly 97 minutes apart, so the data cannot separate these causes. '
        'SPAM-corrected results should remain exploratory. This failure also limits interpretation of the uncorrected evolution, '
        'whose preparation is the same mixture circuit.', '',
        '## Precision, circuit cost, and conclusion', '',
        'Linear CZ extrapolation uses weights (17/6, 1/3, -13/6). Its fixed-path shot SE is about 0.0135–0.0140 per site, '
        'versus about 0.0038–0.0039 without extrapolation: roughly 3.58 times larger. '
        'Three noise levels with small residuals do not establish linearity out to zero noise. Reset/idle effects are not uniformly scaled.', '',
        '| Treatment | CZ factor | Circuits | CZ gates | Resets | Recorded native depth |',
        '|---|---:|---:|---:|---:|---|']
    for m in metrics:
        lines.append(f"| {m['arm']} | {m['factor']} | {m['circuits']} | {m['cz_counts']} | {m['reset_counts']} | {m['depth_min']}–{m['depth_max']} |")
    lines += ['',
        'This is encouraging evidence for reducing the worst population discrepancy, with twirling showing the clearest gain. '
        'It is not a demonstrated 0.02 steady-state result: every method has a point-estimate physical maximum above 0.02, '
        'the ideal 64-path subset itself differs from the physical target by 0.023795, and extrapolation uncertainty is substantial. '
        'These are population-only measurements; excitation currents were not measured.', '',
        'Before a large production run, retain the same first-order schedule and test whether the mixture failure persists '
        'with preparation controls repeated within each job. A paired confirmation of twirling and ZNE needs a separately '
        'chosen sampling budget; increasing only extrapolated shots will not address control failure or drift. '
        'No hardware jobs or backend queries were made for this review.', '']
    (output / 'Findings.md').write_text('\n'.join(lines))
    assert originals == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in run.iterdir() if p.is_file()}
    print(output / 'Findings.md')
    print(json.dumps(serializable(comparisons), indent=2))


def plot_populations(output, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected = [0, 4, 8]
    names = ['raw', 'twirled', 'zne']
    labels = ['Raw', 'Twirled', 'ZNE']
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
    for ax, j in zip(axes, selected):
        points = [report['statistics']['profiles'][name]['populations'][j] for name in names]
        errors = [1.96*report['statistics']['profiles'][name]['fixed_path_delta_method_se'][j] for name in names]
        ax.errorbar(range(3), points, yerr=errors, fmt='o', capsize=4, color='#1565c0', label='Hardware / extrapolated')
        ax.axhline(report['ideal_same_paths'][j], color='#00897b', linestyle='--', label='Ideal: same 64 paths')
        ax.axhline(report['physical_target'][j], color='black', linestyle=':', label='Physical stationary')
        ax.set(xticks=range(3), xticklabels=labels, ylabel='Population',
               title=f"Site {j+1} (physical {report['configuration']['physical_path'][j+1]})")
        ax.grid(alpha=.2); ax.legend(fontsize=7)
    for ext in ('png', 'pdf'):
        fig.savefig(output / f'selected_populations.{ext}', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
