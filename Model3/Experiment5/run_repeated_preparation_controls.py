"""Repeat matched preparation/readout controls inside each IBM job.

Default is an offline plan. Only --execute or --resume submits jobs.
No Trotter steps, bath collisions, or reservoir resets are executed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import norm

DIRECTORY = Path(__file__).resolve().parent
if str(DIRECTORY.parents[1]) not in sys.path:
    sys.path.insert(0, str(DIRECTORY.parents[1]))
from library.reservoir_scaling import atomic_json
from Model3.Experiment5 import mitigation as mit, run_mitigation as pilot
from Model3.Experiment5 import run_hardware as hardware, circuit_depth as circuits
from Model3.Experiment5.analyze_hardware import moments, serializable, source_integrity

EXPERIMENT = 'Model3/Experiment5/repeated_preparation_controls_v1'
DEFAULT_REFERENCE = DIRECTORY/'hardware_runs/ibm_phoenix_mitigation_20261026T184843_413808Z'
PREPARATIONS = ('all_zero', 'all_one', 'initial_mixture')


def validate_settings(cfg):
    if (cfg['jobs'] < 1 or cfg['rounds'] < 2 or cfg['randomizations'] < 2
            or cfg['randomizations'] % 2 or cfg['shots'] < 2*cfg['randomizations']
            or cfg['shots'] % cfg['randomizations'] or min(cfg['seed'], cfg['seed_transpiler']) < 0
            or cfg['optimization_level'] not in range(4) or not 0 < cfg['alpha'] < 1):
        raise ValueError('need jobs >=1, rounds >=2, even randomizations >=2, divisible shots with >=2 per variant, valid seeds/alpha')


def parse_arguments(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run', type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--shots', type=int, default=2048, help='total per preparation/readout treatment/round/job')
    parser.add_argument('--randomizations', type=int, default=4)
    parser.add_argument('--seed', type=int, default=29)
    parser.add_argument('--seed-transpiler', type=int, default=11)
    parser.add_argument('--optimization-level', type=int, choices=range(4), default=3)
    parser.add_argument('--alpha', type=float, default=.05)
    parser.add_argument('--account-file', type=Path)
    parser.add_argument('--output-directory', type=Path, default=DIRECTORY/'hardware_runs')
    parser.add_argument('--analysis-output-directory', type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--plan-only', action='store_true')
    action.add_argument('--compile-only', action='store_true')
    action.add_argument('--execute', action='store_true')
    action.add_argument('--resume', type=Path)
    action.add_argument('--analyze-only', type=Path)
    options = parser.parse_args(arguments)
    if not (options.compile_only or options.execute or options.resume or options.analyze_only):
        options.plan_only = True
    if not (options.resume or options.analyze_only):
        validate_settings(vars(options))
    return options


def plan(cfg):
    validate_settings(cfg)
    rng = np.random.default_rng(cfg['seed'])
    n = len(cfg['physical_path'])-2
    masks = mit.balanced_masks(n, cfg['randomizations'], cfg['seed'])
    records, batches = [], []
    for job in range(cfg['jobs']):
        start = len(records)
        for round_index in range(cfg['rounds']):
            rows = []
            for name, population in zip(PREPARATIONS, (0., 1., .5)):
                variants = [('raw', None, 0, cfg['shots'])]
                variants += [('twirled', v, mask, cfg['shots']//len(masks)) for v, mask in enumerate(masks)]
                for arm, v, mask, shots in variants:
                    rows.append(dict(kind='control', control=name, arm=arm, randomization=v,
                        job_index=job, round_index=round_index, measurement_mask=mask, shots=shots,
                        expected_populations=[population]*n,
                        case_id=f'job{job}_round{round_index}_{arm}_{name}_{v}'))
            for index in rng.permutation(len(rows)):
                records.append(dict(rows[index], pub_index=len(records)))
        batches.append(dict(start=start, stop=len(records), status='not_submitted', job_id=None, compiled=False))
    return records, batches


def verify_manifest(manifest, source):
    if manifest.get('experiment') != EXPERIMENT:
        raise ValueError('not a repeated-preparation manifest')
    cfg = manifest['configuration']
    hardware.require_validated_source(source)
    n = source['configuration']['physics']['n_qubits']
    if (cfg['steps'] != 0 or cfg['sampler_options'] != pilot.SAMPLER_OPTIONS
            or len(cfg['physical_path']) != n+2 or min(cfg['physical_path']) < 0
            or len(set(cfg['physical_path'])) != n+2 or source['preparation']['populations'] != [.5]*n):
        raise ValueError('source/mapping/zero-step protocol mismatch')
    expected, batches = plan(cfg)
    if len(expected) != len(manifest['circuits']) or len(batches) != len(manifest['batches']):
        raise ValueError('incomplete control plan')
    layout = cfg['physical_path'][1:-1]+[cfg['physical_path'][0], cfg['physical_path'][-1]]
    for a, b in zip(expected, manifest['circuits']):
        if any(b.get(k) != v for k, v in a.items()):
            raise ValueError('control cases/shot allocation changed')
        if 'final_logical_to_physical' in b and b['final_logical_to_physical'] != layout:
            raise ValueError('control mapping changed')
        if 'metrics' in b and (b['metrics']['two_qubit_gates'] or b['metrics']['operation_counts'].get('reset', 0)):
            raise ValueError('control unexpectedly contains two-qubit gates or resets')
    for a, b in zip(batches, manifest['batches']):
        if (a['start'], a['stop']) != (b['start'], b['stop']):
            raise ValueError('job coverage changed')
        if b['status'] not in ('not_submitted', 'submitting', 'submitted', 'collected'):
            raise ValueError('unknown job state')
        if b['status'] in ('submitted', 'collected') and not b.get('job_id'):
            raise ValueError('submitted job lacks an ID; reconcile with IBM')
    if cfg['total_circuits'] != len(expected) or cfg['total_shots'] != sum(r['shots'] for r in expected):
        raise ValueError('saved control budget changed')


def read_source(folder, manifest):
    raw = (folder/'source.json').read_bytes()
    integrity = source_integrity(raw, manifest['configuration']['source_sha256'])
    source = json.loads(raw)
    verify_manifest(manifest, source)
    return source, integrity


def summarize_controls(groups, n, cfg, job_index, round_index):
    rows = []
    for arm in ('raw', 'twirled'):
        result = {name: mit.group_moments(groups[arm][name], n) for name in PREPARATIONS}
        a, b, mix = [result[name][0] for name in PREPARATIONS]
        ca, cb, cm = [result[name][1] for name in PREPARATIONS]
        prediction = (a+b)/2
        se = np.sqrt(np.maximum(0, np.diag(cm+(ca+cb)/4)))
        contrast = b-a
        corrected = np.full(n, np.nan)
        corrected_se = np.full(n, np.nan)
        good = contrast >= .5
        corrected[good] = (mix[good]-a[good])/contrast[good]
        variance = (np.diag(cm)+(corrected-1)**2*np.diag(ca)+corrected**2*np.diag(cb))/np.where(good, contrast, 1.)**2
        corrected_se[good] = np.sqrt(np.maximum(0, variance[good]))
        residual = mix-prediction
        z = np.divide(abs(residual), se, out=np.where(abs(residual) > 1e-12, np.inf, 0.), where=se > 0)
        p = 2*norm.sf(z)
        for j in range(n):
            rows.append(dict(job_index=job_index, job_number=job_index+1,
                round_index=round_index, round_number=None if round_index is None else round_index+1,
                scope='job_pooled' if round_index is None else 'within_round',
                arm=arm, site=j+1, physical_qubit=cfg['physical_path'][j+1],
                zero=a[j], one=b[j], mixture=mix[j], predicted_mixture=prediction[j],
                mixture_residual=residual[j], residual_se=se[j], approximate_p=p[j], contrast=contrast[j],
                corrected_mixture=float(corrected[j]) if good[j] else None,
                corrected_se=float(corrected_se[j]) if good[j] else None,
                corrected_ci95=(corrected[j]+np.array([-1., 1.])*1.96*corrected_se[j]).tolist() if good[j] else None))
    return rows


def analyze(folder):
    manifest = json.loads((folder/'manifest.json').read_bytes())
    source, integrity = read_source(folder, manifest)
    if manifest['status'] != 'collected':
        raise ValueError('need all control jobs collected')
    cfg = manifest['configuration']; n = len(cfg['physical_path'])-2
    rows, verified, missing, jobs = [], [], [], []
    total_shots = 0
    for job_index, batch in enumerate(manifest['batches']):
        records = manifest['circuits'][batch['start']:batch['stop']]
        if (batch['status'] != 'collected' or not batch['compiled']
                or len(batch.get('observations', [])) != len(records)):
            raise ValueError('incomplete control job')
        if (folder/batch['isa_file']).exists():
            pilot.compiled_batch(folder, manifest, batch, source); verified.append(batch['isa_file'])
        else:
            missing.append(batch['isa_file'])
        groups = {(round_index, arm, name): [] for round_index in range(cfg['rounds'])
                  for arm in ('raw', 'twirled') for name in PREPARATIONS}
        for record, obs in zip(records, batch['observations']):
            counts = mit.undo_measurement_twirl(obs['outcome_counts'], n, record['measurement_mask'])
            shots = moments(counts, n)[2]
            if shots != record['shots']:
                raise ValueError('control shot mismatch')
            total_shots += shots
            groups[record['round_index'], record['arm'], record['control']].append(counts)
        for round_index in [*range(cfg['rounds']), None]:
            selected = range(cfg['rounds']) if round_index is None else [round_index]
            arm_groups = {arm: {name: sum([groups[r, arm, name] for r in selected], [])
                         for name in PREPARATIONS} for arm in ('raw', 'twirled')}
            for arm in arm_groups:
                expected = len(selected)*(1 if arm == 'raw' else cfg['randomizations'])
                if any(len(g) != expected for g in arm_groups[arm].values()):
                    raise ValueError('missing preparation/readout variant')
            rows.extend(summarize_controls(arm_groups, n, cfg, job_index, round_index))
        jobs.append(dict(job_number=job_index+1, job_id=batch['job_id'], metrics=batch.get('job_metrics'),
            collected_at_utc=batch.get('collected_at_utc')))
    if total_shots != cfg['total_shots']:
        raise ValueError('total control shot mismatch')
    order = np.argsort([r['approximate_p'] for r in rows])
    adjusted = np.minimum(1, np.maximum.accumulate(np.array([rows[i]['approximate_p'] for i in order])*(len(rows)-np.arange(len(rows)))))
    for index, p in zip(order, adjusted):
        rows[index]['holm_p'] = float(p)
        rows[index]['mixture_flagged'] = bool(p < cfg['alpha'])
    return dict(experiment=EXPERIMENT, run_directory=str(folder.resolve()), configuration=cfg, jobs=jobs,
        sites=rows, verification=dict(source_integrity=integrity, compiled_batches_verified=verified,
        compiled_batches_not_available=missing, raw_counts_reconstructed=True),
        scope='Controls only, without evolution. Mixture residual is mixture-(zero+one)/2 within the same round/job. '
        'Holm correction covers all sites, readout treatments, rounds and job-pooled checks; normal approximations, '
        'independent shots and stable local response are assumptions. Passing does not certify evolved-state accuracy. '
        'Within-job requested order is recorded, but actual dispatch order/timing is not guaranteed. '
        'No automatic follow-up hardware execution or mapping selection.')


def save_analysis(report, output):
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if output.resolve().is_relative_to(Path(report['run_directory']).resolve()):
        raise ValueError('analysis must be outside raw run')
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output/'analysis.json', serializable(report))
    with (output/'controls.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report['sites'][0])); writer.writeheader(); writer.writerows(report['sites'])
    cfg = report['configuration']; n = len(cfg['physical_path'])-2
    selected = sorted(set([0, min(2, n-1), max(0, n-2)]))
    fig, axes = plt.subplots(1, len(selected), figsize=(12, 4), squeeze=False, layout='constrained')
    for ax, j in zip(axes.flat, selected):
        for arm, shift in [('raw', -.09), ('twirled', .09)]:
            rows = [r for r in report['sites'] if r['site'] == j+1 and r['arm'] == arm and r['scope'] == 'within_round' and r['corrected_mixture'] is not None]
            x = [r['job_index']*cfg['rounds']+r['round_index']+1+shift for r in rows]
            ax.errorbar(x, [r['corrected_mixture'] for r in rows], yerr=[1.96*r['corrected_se'] for r in rows], fmt='o', capsize=3, label=arm)
        ax.axhline(.5, color='black', linestyle='--', label='Ideal mixture')
        ax.set(title=f"Site {j+1}, physical {cfg['physical_path'][j+1]}", ylabel='Locally corrected mixture population', xlabel='Requested job / round',
            xticks=range(1, cfg['jobs']*cfg['rounds']+1),
            xticklabels=[f'{job+1}/{r+1}' for job in range(cfg['jobs']) for r in range(cfg['rounds'])])
        ax.grid(alpha=.2); ax.legend(fontsize=7)
    for ext in ('png', 'pdf'):
        fig.savefig(output/f'mixture_controls.{ext}', dpi=170)
    plt.close(fig)
    lines = ['# Repeated preparation controls', '', report['scope'], '',
        '![Selected control populations, ±1.96 conditional SE](mixture_controls.png)', '',
        '| Job | Arm | Site | Physical | Zero | One | Mixture | Predicted | Corrected | Flagged |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in report['sites']:
        if r['scope'] == 'job_pooled':
            corrected = 'unreliable contrast' if r['corrected_mixture'] is None else f"{r['corrected_mixture']:.6f}"
            lines.append(f"| {r['job_number']} | {r['arm']} | {r['site']} | {r['physical_qubit']} | {r['zero']:.6f} | {r['one']:.6f} | {r['mixture']:.6f} | {r['predicted_mixture']:.6f} | {corrected} | {r['mixture_flagged']} |")
    lines += ['', 'All round-level data, multiplicity-adjusted approximate tests, contrasts, and uncertainties are in controls.csv and analysis.json. '
        'A flag identifies inconsistency with the local response model; it does not uniquely identify preparation, readout, latency or drift as its cause. '
        'A failed contrast is recorded without inversion or clipping. No large evolution run is automatically started.', '']
    (output/'Analysis.md').write_text('\n'.join(lines))


def main(options, *, backend=None, service=None, sampler_factory=None):
    if options.analyze_only:
        report = analyze(options.analyze_only)
        output = options.analysis_output_directory or DIRECTORY/'analysis/hardware'/options.analyze_only.name/'repeated_preparation'
        save_analysis(report, output); print(output/'Analysis.md'); return report
    if options.resume:
        folder = options.resume.resolve().parent
        manifest = json.loads(options.resume.read_bytes()); source, _ = read_source(folder, manifest)
    else:
        reference = options.reference_run
        old = json.loads((reference/'manifest.json').read_bytes())
        source, _ = pilot.read_source(reference, old)
        source_raw = (reference/'source.json').read_bytes()
        cfg = {key: getattr(options, key) for key in ('jobs', 'rounds', 'shots', 'randomizations', 'seed', 'seed_transpiler', 'optimization_level', 'alpha')}
        cfg.update(backend=old['configuration']['backend'], physical_path=old['configuration']['physical_path'], steps=0,
            sampler_options=pilot.SAMPLER_OPTIONS, reference_run=reference.name,
            reference_manifest_sha256=pilot.sha((reference/'manifest.json').read_bytes()), source_sha256=pilot.sha(source_raw),
            software_versions={key: importlib.metadata.version(key) for key in ('qiskit', 'qiskit-ibm-runtime', 'numpy')})
        records, batches = plan(cfg)
        cfg.update(total_circuits=len(records), total_shots=sum(r['shots'] for r in records))
        manifest = dict(experiment=EXPERIMENT, configuration=cfg, circuits=records, batches=batches, status='planned')
        verify_manifest(manifest, source)
        folder = options.output_directory/f"{cfg['backend']}_repeated_preparation_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')}"
        folder.mkdir(parents=True, exist_ok=False)
        (folder/'source.json').write_bytes(source_raw); atomic_json(folder/'manifest.json', manifest)
    cfg = manifest['configuration']
    print(f"Controls plan: {cfg['total_circuits']} circuits, {cfg['total_shots']} shots, {cfg['jobs']} jobs. {folder}", flush=True)
    if options.plan_only:
        print('Offline plan only; no account/backend access or hardware submission. Use --execute for a fresh run or --resume this manifest.', flush=True)
        return manifest
    if any(b['status'] == 'submitting' and not b.get('job_id') for b in manifest['batches']):
        raise ValueError('submission has no recorded job ID; reconcile with IBM; no automatic retry')
    if all(b['status'] == 'collected' for b in manifest['batches']):
        report = analyze(folder)
        output = options.analysis_output_directory or DIRECTORY/'analysis/hardware'/folder.name/'repeated_preparation'
        save_analysis(report, output); print(output/'Analysis.md'); return report
    sampler = None
    for job_index, batch in enumerate(manifest['batches']):
        if batch['status'] == 'collected':
            continue
        if batch.get('job_id'):
            if options.compile_only:
                continue
            service = service or hardware.make_service(options)
            job = service.job(batch['job_id'])
        else:
            if backend is None:
                service = service or hardware.make_service(options)
                backend = service.backend(cfg['backend'], use_fractional_gates=False)
            if backend.name != cfg['backend']:
                raise ValueError('backend differs from reference')
            refresh = getattr(backend, 'refresh', None)
            if callable(refresh): refresh()
            circuits.chain_layout(backend, len(cfg['physical_path'])-2, 'boundary', cfg['physical_path'])
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
            snapshot_dir = folder/'calibration'/f'job{job_index+1}_{stamp}'
            snapshot_dir.mkdir(parents=True, exist_ok=False)
            hardware.save_backend_snapshot(backend, cfg['physical_path'], manifest['circuits'], snapshot_dir)
            snapshot = snapshot_dir/'backend_snapshot.json'
            manifest.setdefault('calibration_snapshots', []).append(dict(job_index=job_index,
                file=snapshot.relative_to(folder).as_posix(), sha256=pilot.sha(snapshot.read_bytes())))
            atomic_json(folder/'manifest.json', manifest)
            # Uses exactly the prior pilot's preparation, transform and native
            # verification; records are controls, so its body always has 0 steps.
            pubs = pilot.compiled_batch(folder, manifest, batch, source, backend)
            verify_manifest(manifest, source)
            if options.compile_only: continue
            if sampler is None:
                if sampler_factory is None:
                    from qiskit_ibm_runtime import SamplerV2
                    sampler_factory = SamplerV2
                sampler = sampler_factory(mode=backend, options=pilot.SAMPLER_OPTIONS)
            records = manifest['circuits'][batch['start']:batch['stop']]
            batch['status'] = 'submitting'; atomic_json(folder/'manifest.json', manifest)
            job = sampler.run([(pub, None, r['shots']) for pub, r in zip(pubs, records)], shots=cfg['shots'])
            job_id = job.job_id()
            if not isinstance(job_id, str) or not job_id.strip():
                raise ValueError('submission returned no ID; reconcile with IBM before resuming')
            batch.update(job_id=job_id, status='submitted'); atomic_json(folder/'manifest.json', manifest)
            print(f'Submitted control job {job_index+1}: {job_id}', flush=True)
        records = manifest['circuits'][batch['start']:batch['stop']]
        result = job.result()
        if len(result) != len(records): raise ValueError('Runtime result length mismatch')
        observations = []
        for pub, record in zip(result, records):
            counts = {str(k): int(v) for k, v in pub.data.outcome.get_counts().items()}
            if moments(counts, len(cfg['physical_path'])-2)[2] != record['shots']:
                raise ValueError('Runtime shot mismatch')
            observations.append(dict(outcome_counts=counts))
        batch.update(status='collected', observations=observations, collected_at_utc=datetime.now(timezone.utc).isoformat())
        try: batch['job_metrics'] = json.loads(json.dumps(job.metrics(), default=str))
        except Exception as error: batch['job_metrics_unavailable'] = str(error)
        atomic_json(folder/'manifest.json', manifest)
    if options.compile_only:
        manifest['status'] = 'compiled'; atomic_json(folder/'manifest.json', manifest); bundle(folder)
        print(f'Compiled only, no jobs submitted. Resume: {folder/"manifest.json"}', flush=True); return manifest
    manifest['status'] = 'collected'; atomic_json(folder/'manifest.json', manifest)
    report = analyze(folder); atomic_json(folder/'hardware_results.json', serializable(report))
    output = options.analysis_output_directory or DIRECTORY/'analysis/hardware'/folder.name/'repeated_preparation'
    save_analysis(report, output); bundle(folder, output)
    print(f"Analysis: {output/'Analysis.md'}\nTransfer: {folder/'results_bundle.zip'}", flush=True)
    return report


def bundle(folder, output=None):
    import zipfile
    with zipfile.ZipFile(folder/'results_bundle.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in ('manifest.json', 'source.json', 'hardware_results.json'):
            if (folder/name).exists(): archive.write(folder/name, name)
        for path in folder.glob('calibration/*/backend_snapshot.json'):
            archive.write(path, path.relative_to(folder).as_posix())
        if output:
            for path in output.iterdir():
                if path.is_file(): archive.write(path, 'analysis/'+path.name)


if __name__ == '__main__':
    try: main(parse_arguments())
    except (ValueError, OSError) as error: raise SystemExit(str(error)) from error
