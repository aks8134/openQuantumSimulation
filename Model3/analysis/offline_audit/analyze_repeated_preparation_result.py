"""Review pulled repeated preparation controls without any backend access."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.stats import norm

from Model3.Experiment5 import run_repeated_preparation_controls as runner
from Model3.Experiment5.analyze_hardware import serializable
from library.reservoir_scaling import atomic_json

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN = ROOT/'Model3/Experiment5/hardware_runs/ibm_phoenix_repeated_preparation_20261027T043444_434996Z'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-directory', type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    folder = args.run_directory.resolve()
    output = ROOT/'Model3/Experiment5/analysis/hardware'/folder.name/'preparation_review'
    originals = {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in folder.rglob('*') if p.is_file()}
    manifest = json.loads((folder/'manifest.json').read_bytes())
    cached = json.loads((folder/'hardware_results.json').read_bytes())
    report = runner.analyze(folder)
    # Windows/Linux SciPy tail probabilities can differ in the final bits.
    # Require exact populations; allow rounding only in derived statistics.
    assert len(report['sites']) == len(cached['sites'])
    for row, old in zip(report['sites'], cached['sites'], strict=True):
        for key, value in serializable(row).items():
            if isinstance(value, (float, list)):
                np.testing.assert_allclose(value, old[key], atol=1e-14, rtol=1e-12)
            else:
                assert value == old[key], key
        for key in ('zero', 'one', 'mixture', 'predicted_mixture'):
            assert row[key] == old[key]
    cfg = report['configuration']
    reference = folder.parent/cfg['reference_run']
    # Git normalizes Windows CRLF JSON files. The transfer archive retains the
    # original bytes against which the execution machine recorded hashes.
    with zipfile.ZipFile(reference/'results_bundle.zip') as archive:
        raw_reference = archive.read('manifest.json')
        assert hashlib.sha256(raw_reference).hexdigest() == cfg['reference_manifest_sha256']
        assert json.loads(raw_reference) == json.loads((reference/'manifest.json').read_bytes())
    with zipfile.ZipFile(folder/'results_bundle.zip') as archive:
        assert archive.testzip() is None
        for name in ('manifest.json', 'source.json', 'hardware_results.json'):
            assert json.loads(archive.read(name)) == json.loads((folder/name).read_bytes())
        for item in manifest['calibration_snapshots']:
            path = folder/item['file']
            raw_snapshot = archive.read(item['file'])
            assert hashlib.sha256(raw_snapshot).hexdigest() == item['sha256']
            assert json.loads(raw_snapshot) == json.loads(path.read_bytes())
    runner.save_analysis(report, output)

    variants = []
    for job, batch in enumerate(manifest['batches']):
        for record, observation in zip(manifest['circuits'][batch['start']:batch['stop']], batch['observations'], strict=True):
            assert record['metrics']['two_qubit_gates'] == 0
            assert record['metrics']['operation_counts'].get('reset', 0) == 0
            counts = runner.mit.undo_measurement_twirl(observation['outcome_counts'], 10, record['measurement_mask'])
            means, _, shots = runner.moments(counts, 10)
            if record['control'] == 'initial_mixture':
                for j in (2, 7, 8):
                    variants.append(dict(job=job+1, round=record['round_index']+1, arm=record['arm'],
                        variant=record['randomization'], pub_index=record['pub_index'], site=j+1,
                        physical_qubit=cfg['physical_path'][j+1], terminal_flip=(record['measurement_mask'] >> j) & 1,
                        decoded_mixture=means[j], shots=shots))
    with (output/'mixture_variants.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(variants[0])); writer.writeheader(); writer.writerows(variants)
    pooled = [r for r in report['sites'] if r['scope'] == 'job_pooled']
    changes = []
    for arm in ('raw', 'twirled'):
        for site in range(1, 11):
            a, b = [next(r for r in pooled if r['job_number'] == job and r['arm'] == arm and r['site'] == site) for job in (1, 2)]
            delta = b['mixture_residual']-a['mixture_residual']
            se = np.hypot(a['residual_se'], b['residual_se'])
            changes.append(dict(arm=arm, site=site, physical_qubit=a['physical_qubit'],
                job2_minus_job1_residual=delta, se=se,
                approximate_p=2*norm.sf(abs(delta)/se) if se else (0. if delta else 1.)))
    order = np.argsort([r['approximate_p'] for r in changes])
    adjusted = np.minimum(1, np.maximum.accumulate(np.array([changes[i]['approximate_p'] for i in order])*(len(changes)-np.arange(len(changes)))))
    for index, p in zip(order, adjusted): changes[index]['holm_p_20_changes'] = p
    old_snapshot = json.loads((reference/'backend_snapshot.json').read_bytes())
    new_snapshots = [json.loads((folder/item['file']).read_bytes()) for item in manifest['calibration_snapshots']]
    calibration = {str(q): dict(prior_t1_us=old_snapshot['qubits'][str(q)]['t1']*1e6,
        new_t1_us=[snap['qubits'][str(q)]['t1']*1e6 for snap in new_snapshots]) for q in (48, 76, 86)}
    review = dict(reconstruction_matches_saved_sites=True, transfer_bundle_verified=True,
        reference_manifest_verified=True, verification=report['verification'],
        flagged_rows=[r for r in report['sites'] if r['mixture_flagged']],
        between_job_changes=changes, calibration_comparison=calibration, input_sha256=originals,
        scope='Approximate independent-shot tests, conditional on recorded variants and grouping. '
        'Multiple checks and repeated hardware contexts do not provide independent causal experiments.')
    atomic_json(output/'review.json', serializable(review))
    plot_selected(report, output)
    lines = ['# Review of repeated preparation controls', '',
        'Both jobs completed, with 60 circuits and 49,152 shots. Reconstructing all final counts and terminal masks '
        'reproduces every saved population exactly and derived statistics to rounding precision. The source, reference manifest, calibration snapshots and transfer '
        'archive pass integrity checks. Native QPY was not transferred, so native contents cannot be independently '
        'rechecked here; the execution-machine report records both native batches as verified.', '',
        '![Locally corrected mixtures at three sites](selected_mixture_controls.png)', '',
        'Error bars are 1.96 times conditional shot/calibration SE, not simultaneous bounds and not a drift model.', '',
        '## Qubit 48: the previous large deficit did not recur', '',
        'The previous mitigation pilot measured 0.4150 raw / 0.4375 terminal-twirled at site 3 (physical 48). '
        'This run gives the following job-pooled mixtures, each calibrated using its own job:', '',
        '| Job | Readout | Observed mixture | Zero/one prediction | Corrected mixture | Conditional 95% interval |',
        '|---:|---|---:|---:|---:|---|']
    for r in pooled:
        if r['site'] == 3:
            lo, hi = r['corrected_ci95']
            lines.append(f"| {r['job_number']} | {r['arm']} | {r['mixture']:.6f} | {r['predicted_mixture']:.6f} | {r['corrected_mixture']:.6f} | [{lo:.6f}, {hi:.6f}] |")
    lines += ['', 'No qubit-48 round or pooled check is flagged. The data do not support a persistent, large deficit '
        'on that qubit. They also do not establish whether the earlier failure was drift, context, or another transient effect.', '',
        '## Qubit 76: a different within-job failure', '',
        '| Job / round | Readout | Observed mixture | Zero/one prediction | Corrected mixture | Holm-adjusted p |',
        '|---|---|---:|---:|---:|---:|']
    for r in report['sites']:
        if r['site'] == 8 and r['arm'] == 'twirled':
            label = f"{r['job_number']} / {r['round_number'] if r['round_number'] else 'pooled'}"
            lines.append(f"| {label} | {r['arm']} | {r['mixture']:.6f} | {r['predicted_mixture']:.6f} | {r['corrected_mixture']:.6f} | {r['holm_p']:.3g} |")
    bad = next(r for r in pooled if r['site'] == 8 and r['arm'] == 'twirled' and r['job_number'] == 2)
    lo, hi = bad['corrected_ci95']
    lines += ['', f"Job 2's corrected twirled mixture is {bad['corrected_mixture']:.6f}, interval [{lo:.6f}, {hi:.6f}], "
        f"rather than 0.5. Its two rounds and pooled result are the only three flagged rows among 120 adjusted checks. "
        'These are repeated views of the same failure, not three independent hardware events. All 20 distinct site/readout '
        'combinations are otherwise unflagged at the predeclared multiplicity-adjusted threshold; absence of a flag is not '
        'an equivalence or 0.02-accuracy certificate.', '',
        'All eight job-2 twirled mixture variants at qubit 76 lie between 0.4102 and 0.4590 after decoding. '
        'The deficit occurs with and without a terminal X, and in both rounds. It is not explained by one anomalous variant '
        'or solely by transferring zero/one calibration from a different job. Dependence on other flipped sites or actual '
        'dispatch timing remains possible. Job-2 raw mixture is 0.4719 (corrected 0.4778), a smaller deficit whose within-job '
        'test does not survive the full family adjustment.', '',
        'The twirled residual changes from -0.010010 in job 1 to -0.056519 in job 2. '
        'Between-job change tests and all individual variant values are saved in review.json and mixture_variants.csv. '
        'Changing output across jobs is evidence of context/time sensitivity, not proof of one specific drift mechanism.', '',
        '## Calibration and implications', '',
        f"Qubit 76's saved T1 is {calibration['76']['new_t1_us'][0]:.1f} microseconds in both new snapshots, "
        f"down from {calibration['76']['prior_t1_us']:.1f} microseconds in the mitigation pilot. "
        'This is a relevant coherence vulnerability and is below the earlier evolution-chain T1 screen of 80 microseconds. '
        'The controls runner deliberately retained the reference mapping rather than filtering it out. Both new snapshots '
        'contain identical qubit-76 calibration, so this recorded T1 change does not explain why job 2 differs from job 1. '
        'It is not a measured duration model or proof that T1 decay caused the deficit.', '',
        'The experiment resolves the cross-job calibration ambiguity for the new failure: a mixture inconsistency remains '
        'even when its zero/one controls come from the same round/job. It occurs with no CZ gates and no reservoir resets, '
        'so deep evolution and bath-reset errors cannot be its only possible origin. Initialization, intermediate preparation '
        'measurement and timing, measurement context, and temporal hardware effects remain relevant. Preparation-register '
        'counts were not saved in this protocol, so the preparation outcome and subsequent final readout cannot be compared shot by shot.', '',
        'The practical conclusion is to keep the cosine schedule fixed and avoid treating the chain or terminal-twirled '
        'readout as universally validated. Replacing qubit 48 alone is not supported by this run. Any production mitigation '
        'evaluation should keep same-job preparation controls alongside evolution and assess the current chain calibration; '
        'passing these short controls would still not establish deep-circuit accuracy. No new hardware run was launched for this review.', '']
    (output/'Findings.md').write_text('\n'.join(lines))
    assert originals == {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in folder.rglob('*') if p.is_file()}
    print(output/'Findings.md')
    print(json.dumps(serializable(review['flagged_rows']), indent=2))
    print('Qubit 76 between-job changes:', [serializable(r) for r in changes if r['site'] == 8])


def plot_selected(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
    for ax, site in zip(axes, (3, 8, 9)):
        for arm, offset in [('raw', -.08), ('twirled', .08)]:
            rows = [r for r in report['sites'] if r['site'] == site and r['arm'] == arm and r['scope'] == 'within_round']
            x = [r['job_index']*2+r['round_index']+1+offset for r in rows]
            ax.errorbar(x, [r['corrected_mixture'] for r in rows],
                yerr=[1.96*r['corrected_se'] for r in rows], fmt='o', capsize=3, label=arm)
        ax.axhline(.5, color='black', linestyle='--', label='Expected mixture')
        ax.axvline(2.5, color='grey', linewidth=.8, alpha=.5)
        ax.set(title=f"Site {site}, physical {report['configuration']['physical_path'][site]}",
            xlabel='Job / round', ylabel='Locally corrected mixture population',
            xticks=range(1, 5), xticklabels=['1/1', '1/2', '2/1', '2/2'])
        ax.grid(alpha=.2); ax.legend(fontsize=7)
    for ext in ('png', 'pdf'): fig.savefig(output/f'selected_mixture_controls.{ext}', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
