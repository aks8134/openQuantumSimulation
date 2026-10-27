"""Repeated control plans, native equivalence, analysis and mocked recovery."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

import numpy as np

from Model3.Experiment5 import run_repeated_preparation_controls as runner
from Model3.Experiment5 import run_hardware as hardware
from tests import test_model3_mitigation as mitigation_fixtures
from tests.test_model3_experiment5_hardware import pub


class RepeatedPreparationTests(unittest.TestCase):
    def options(self, folder, *extra):
        pilot_options, backend = mitigation_fixtures.MitigationTests().options(folder, '--plan-only')
        runner.pilot.main(pilot_options)
        reference = next((Path(folder)/'mitigation').iterdir())
        return runner.parse_arguments(['--reference-run', str(reference), '--shots', '128',
            '--randomizations', '2', '--output-directory', str(Path(folder)/'controls'),
            '--analysis-output-directory', str(Path(folder)/'analysis'), *extra]), backend

    def sampler(self, *, fail_mixture=False):
        sampler = Mock(); jobs = []
        def submit(specs, shots):
            results = []
            for circuit, _, count in specs:
                key = circuit.metadata['mitigation_case_id']
                mask = circuit.metadata.get('mitigation', {}).get('measurement_mask', 0)
                zero, one = format(mask, '03b'), format(mask ^ 7, '03b')
                if 'all_zero' in key: counts = {zero: count}
                elif 'all_one' in key: counts = {one: count}
                elif fail_mixture:
                    # Site 3 is prepared as zero in this deliberately bad mixture.
                    alternate = format(mask ^ 3, '03b')
                    counts = {zero: count//2, alternate: count-count//2}
                else: counts = {zero: count//2, one: count-count//2}
                results.append(pub(counts))
            job = Mock(); job.job_id.return_value = f'offline_controls_{len(jobs)}'
            job.result.return_value = results; job.metrics.return_value = {}
            jobs.append(job); return job
        sampler.run.side_effect = submit
        return sampler, jobs

    def test_default_offline_plan_has_matched_calibration_in_every_round_job(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), \
                patch.object(hardware, 'make_service', side_effect=AssertionError('no account')):
            options = runner.parse_arguments(['--output-directory', folder])
            manifest = runner.main(options)
            cfg = manifest['configuration']
            self.assertTrue(options.plan_only)
            self.assertEqual((cfg['total_circuits'], cfg['total_shots']), (60, 49152))
            self.assertEqual([b['stop']-b['start'] for b in manifest['batches']], [30, 30])
            for job in range(2):
                for round_index in range(2):
                    rows = [r for r in manifest['circuits'] if r['job_index'] == job and r['round_index'] == round_index]
                    for preparation in runner.PREPARATIONS:
                        for arm in ('raw', 'twirled'):
                            selected = [r for r in rows if r['control'] == preparation and r['arm'] == arm]
                            self.assertEqual(sum(r['shots'] for r in selected), 2048)
                            if arm == 'twirled':
                                for j in range(10): self.assertEqual(sum((r['measurement_mask'] >> j) & 1 for r in selected), 2)
            self.assertFalse(list(Path(folder).glob('**/backend_snapshot.json')))
            for arguments in (['--rounds', '1'], ['--jobs', '0'], ['--randomizations', '3'], ['--shots', '7']):
                with self.assertRaises(ValueError): runner.parse_arguments(arguments)

    def test_compile_only_density_equivalence_and_no_evolution(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), \
                patch.object(hardware, 'make_service', side_effect=AssertionError('no account')):
            options, backend = self.options(folder, '--compile-only')
            forbidden = Mock(side_effect=AssertionError('no submission'))
            manifest = runner.main(options, backend=backend, sampler_factory=forbidden)
            forbidden.assert_not_called(); self.assertEqual(manifest['status'], 'compiled')
            directory = next((Path(folder)/'controls').iterdir())
            source, _ = runner.read_source(directory, manifest)
            batch = manifest['batches'][0]
            circuits = runner.pilot.compiled_batch(directory, manifest, batch, source)
            for circuit, record in zip(circuits, manifest['circuits'][:batch['stop']]):
                self.assertEqual(circuit.count_ops().get('cz', 0), 0)
                self.assertEqual(circuit.count_ops().get('reset', 0), 0)
                self.assertEqual(record['final_logical_to_physical'], [1, 2, 3, 0, 4])
                distribution = mitigation_fixtures.probabilities(circuit, [1, 2, 3])
                # Undo only the final measurement mask, never preparation bits.
                decoded = distribution[np.arange(8) ^ record['measurement_mask']]
                if record['control'] == 'initial_mixture': np.testing.assert_allclose(decoded, np.full(8, .125), atol=1e-14)
                else:
                    expected = np.zeros(8); expected[0 if record['control'] == 'all_zero' else 7] = 1
                    np.testing.assert_allclose(decoded, expected, atol=1e-14)

    def test_collection_analysis_portable_bundle_and_completed_resume_is_offline(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), \
                patch.object(hardware, 'make_service', side_effect=AssertionError('no account')):
            options, backend = self.options(folder, '--execute'); sampler, jobs = self.sampler()
            report = runner.main(options, backend=backend, sampler_factory=Mock(return_value=sampler))
            self.assertEqual(len(jobs), 2)
            self.assertEqual(len(report['sites']), 36)  # 2 jobs x (2 rounds + pooled) x 2 arms x 3 sites
            self.assertFalse(any(r['mixture_flagged'] for r in report['sites']))
            for row in report['sites']: self.assertEqual(row['corrected_mixture'], .5)
            directory = Path(report['run_directory'])
            with zipfile.ZipFile(directory/'results_bundle.zip') as archive:
                self.assertIn('analysis/mixture_controls.png', archive.namelist())
                self.assertEqual(sum(name.endswith('backend_snapshot.json') for name in archive.namelist()), 2)
                self.assertFalse(any(name.endswith('.qpy') for name in archive.namelist()))
            original = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
            forbidden = Mock(side_effect=AssertionError('no resubmission'))
            resume = runner.parse_arguments(['--resume', str(directory/'manifest.json'), '--analysis-output-directory', str(Path(folder)/'resumed')])
            runner.main(resume, sampler_factory=forbidden); forbidden.assert_not_called()
            for name, raw in original.items(): self.assertEqual((directory/name).read_bytes(), raw)
            for p in directory.glob('*.qpy'): p.unlink()
            portable = runner.analyze(directory)
            self.assertEqual(len(portable['verification']['compiled_batches_not_available']), 2)

    def test_bad_mixture_is_flagged_with_local_calibration_without_reference_fitting(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), \
                patch.object(hardware, 'make_service', side_effect=AssertionError('no account')):
            options, backend = self.options(folder, '--execute'); sampler, _ = self.sampler(fail_mixture=True)
            report = runner.main(options, backend=backend, sampler_factory=Mock(return_value=sampler))
            for row in report['sites']:
                self.assertEqual(row['mixture_flagged'], row['site'] == 3)
                if row['site'] == 3:
                    self.assertEqual(row['predicted_mixture'], .5)
                    self.assertEqual(row['corrected_mixture'], 0.)
            directory = Path(report['run_directory']); manifest = json.loads((directory/'manifest.json').read_bytes())
            bad = copy.deepcopy(manifest); bad['circuits'][0]['shots'] += 1
            source, _ = runner.read_source(directory, manifest)
            with self.assertRaisesRegex(ValueError, 'allocation'): runner.verify_manifest(bad, source)
            bad = copy.deepcopy(manifest); bad['circuits'][0]['metrics']['operation_counts']['reset'] = 1
            with self.assertRaisesRegex(ValueError, 'resets'): runner.verify_manifest(bad, source)
            original_source = (directory/'source.json').read_bytes()
            (directory/'source.json').write_bytes(original_source.replace(b'0.9', b'0.8', 1))
            with self.assertRaises(ValueError): runner.read_source(directory, manifest)

    def test_interrupted_collection_recovers_known_job_and_submits_only_remaining(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), \
                patch.object(hardware, 'make_service', side_effect=AssertionError('no account')):
            options, backend = self.options(folder, '--execute'); sampler, jobs = self.sampler()
            submit = sampler.run.side_effect
            def interrupt(specs, shots):
                job = submit(specs, shots); job.result.side_effect = RuntimeError('interrupted'); return job
            sampler.run.side_effect = interrupt
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                runner.main(options, backend=backend, sampler_factory=Mock(return_value=sampler))
            manifest_file = next((Path(folder)/'controls').glob('*/manifest.json'))
            jobs[0].result.side_effect = None; sampler.run.side_effect = submit
            service = Mock(); service.job.return_value = jobs[0]
            resume = runner.parse_arguments(['--resume', str(manifest_file), '--analysis-output-directory', str(Path(folder)/'analysis')])
            runner.main(resume, backend=backend, service=service, sampler_factory=Mock(return_value=sampler))
            service.job.assert_called_once_with('offline_controls_0'); self.assertEqual(len(jobs), 2)

    def test_ambiguous_submission_blocks_before_account_access(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), \
                patch.object(hardware, 'make_service', side_effect=AssertionError('no account')):
            options, backend = self.options(folder, '--execute'); sampler = Mock()
            sampler.run.side_effect = RuntimeError('submission response lost')
            with self.assertRaisesRegex(RuntimeError, 'response lost'):
                runner.main(options, backend=backend, sampler_factory=Mock(return_value=sampler))
            manifest_file = next((Path(folder)/'controls').glob('*/manifest.json'))
            resume = runner.parse_arguments(['--resume', str(manifest_file)])
            with self.assertRaisesRegex(ValueError, 'no recorded job ID'): runner.main(resume)


if __name__ == '__main__':
    unittest.main()
