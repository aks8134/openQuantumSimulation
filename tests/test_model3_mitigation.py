"""Mitigation channel equivalence, calibration statistics and mocked lifecycle."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix,Kraus,Operator

from Model3.Experiment5 import mitigation as mit,run_mitigation as runner,run_hardware as hardware
from Model3.Experiment5.analyze_hardware import moments
from tests import test_model3_schedule_comparison as comparison_fixtures
from tests.test_model3_experiment5_hardware import pub as fake_pub


def probabilities(circuit,sites):
    rho=DensityMatrix.from_label('0'*circuit.num_qubits)
    dephase=Kraus([np.diag([1,0]),np.diag([0,1])])
    for item in circuit.data:
        op=item.operation;args=[circuit.find_bit(q).index for q in item.qubits]
        if op.name=='barrier':continue
        if op.name=='measure':rho=rho.evolve(dephase,args)
        elif op.name=='reset':rho=rho.reset(args)
        else:rho=rho.evolve(op,args)
    return rho.probabilities(qargs=sites)


class MitigationTests(unittest.TestCase):
    def options(self,folder,*extra):
        original,backend=comparison_fixtures.ComparisonTests().setup(folder)
        args=['--input-bundle',str(original.input_bundle),'--backend',backend.name,'--physical-path','0','1','2','3','4',
            '--trajectory-count','2','--shots','8','--control-shots','8','--randomizations','2','--batch-size','7',
            '--bootstrap','100','--output-directory',str(Path(folder)/'mitigation'),
            '--analysis-output-directory',str(Path(folder)/'analysis'),*extra]
        return runner.parse_arguments(args),backend

    def sampler(self):
        sampler=Mock();jobs=[]
        def submit(pubs,shots):
            result=[]
            for circuit,_,shots in pubs:
                key=circuit.metadata['mitigation_case_id'];mask=circuit.metadata.get('mitigation',{}).get('measurement_mask',0)
                zero=format(mask,'03b');one=format(mask^7,'03b')
                counts={zero:shots} if 'all_zero' in key else {one:shots} if 'all_one' in key else {zero:shots//2,one:shots//2}
                result.append(fake_pub(counts))
            job=Mock();job.job_id.return_value=f'local_mitigation_{len(jobs)}';job.result.return_value=result;job.metrics.return_value={}
            jobs.append(job);return job
        sampler.run.side_effect=submit;return sampler,jobs

    def test_all_pauli_frames_preserve_cz_unitary(self):
        ideal=QuantumCircuit(2);ideal.cz(0,1)
        for bits in np.ndindex(2,2,2,2):
            with self.subTest(bits=bits):
                circuit=QuantumCircuit(2);mit.twirl_cz(circuit,[0,1],bits)
                self.assertTrue(Operator(circuit).equiv(Operator(ideal)))

    def test_masks_decode_counts_and_are_balanced(self):
        masks=mit.balanced_masks(10,4,29)
        for j in range(10):self.assertEqual(sum((m>>j)&1 for m in masks),2)
        counts={'001':3,'101':5};mask=6
        encoded=mit.undo_measurement_twirl(counts,3,mask)
        self.assertEqual(mit.undo_measurement_twirl(encoded,3,mask),counts)
        for bad in (-1,8):
            with self.assertRaises(ValueError):mit.undo_measurement_twirl(counts,3,bad)

    def test_calibration_inverse_covariance_and_fixed_linear_intercept(self):
        a=np.array([.1]);b=np.array([.8]);population=np.array([.31])
        np.testing.assert_allclose(mit.corrected_population(population,a,b),.3)
        cov=mit.response_covariance(population,np.array([[.002]]),a,np.array([[.001]]),b,np.array([[.0015]]))
        expected=(.002+(.3-1)**2*.001+.3**2*.0015)/.7**2
        self.assertAlmostEqual(cov[0,0],expected)
        x=np.array([1.,1.25,1.5]);w=mit.zne_weights(x)
        self.assertAlmostEqual(w@(0.3+.07*x),.3)
        self.assertGreater(np.linalg.norm(w),3)
        with self.assertRaises(ValueError):mit.corrected_population(population,np.array([.3]),np.array([.4]))
        with self.assertRaises(ValueError):mit.zne_weights([1,1,2])
        self.assertGreater(mit.corrected_population(np.array([.95]),a,b)[0],1)  # no artificial clipping

    def test_default_plan_is_offline_and_exact_workload(self):
        a=runner.parse_arguments([]);self.assertTrue(a.plan_only);self.assertEqual(a.trajectory_count,64)
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()), \
             patch.object(hardware,'make_service',side_effect=AssertionError('no account')):
            a=runner.parse_arguments(['--output-directory',folder]);manifest=runner.main(a)
            self.assertEqual(manifest['status'],'planned')
            self.assertEqual(manifest['configuration']['total_circuits'],847)
            self.assertEqual(manifest['configuration']['total_shots'],77824)
            self.assertEqual([r['pub_index'] for r in manifest['circuits']],list(range(847)))
            self.assertEqual(len(manifest['batches']),2)
            self.assertFalse(any((p/'backend_snapshot.json').exists() for p in Path(folder).iterdir()))
            for index in range(64):
                containing=[b for b in manifest['batches'] if any(r.get('trajectory')==index for r in manifest['circuits'][b['start']:b['stop']])]
                self.assertEqual(len(containing),1)
        for args in (['--randomizations','3'],['--shots','7'],['--batch-size','12'],['--fold-factors','1','1.5'],['--fold-factors','1','2','1.5']):
            with self.subTest(args=args),self.assertRaises(ValueError):runner.parse_arguments(args)

    def test_full_native_density_equivalence_folds_resets_and_measurement_masks(self):
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()), \
             patch.object(hardware,'make_service',side_effect=AssertionError('no account')):
            a,backend=self.options(folder,'--compile-only');forbidden=Mock(side_effect=AssertionError('no job'))
            manifest=runner.main(a,backend=backend,sampler_factory=forbidden);forbidden.assert_not_called()
            directory=next((Path(folder)/'mitigation').iterdir());source,_=runner.read_source(directory,manifest)
            batch=next(b for b in manifest['batches'] if any(r['kind']=='trajectory' for r in manifest['circuits'][b['start']:b['stop']]))
            pubs=runner.compiled_batch(directory,manifest,batch,source)
            records=manifest['circuits'][batch['start']:batch['stop']]
            raw=next(p for p,r in zip(pubs,records) if r['arm']=='raw');sites=raw.metadata['physical_sites']
            expected=probabilities(raw,sites);base=raw.count_ops()['cz'];subsets=[]
            for circuit,record in zip(pubs,records):
                actual=probabilities(circuit,sites);mask=record['measurement_mask']
                np.testing.assert_allclose(actual[np.arange(8)^mask],expected,atol=4e-13)
                self.assertEqual(circuit.count_ops()['reset'],1)
                if record['arm']=='twirled':
                    self.assertEqual(circuit.count_ops()['cz'],base+2*record['folds'])
                    if record['randomization']==0:subsets.append((record['folds'],set(circuit.metadata['mitigation']['folded_cz_indices'])))
            subsets.sort()
            for (_,a),(_,b) in zip(subsets,subsets[1:]):self.assertTrue(a<=b)

    def test_mock_collection_portable_analysis_and_no_resubmission(self):
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()), \
             patch.object(hardware,'make_service',side_effect=AssertionError('no account')):
            a,backend=self.options(folder,'--execute');sampler,jobs=self.sampler()
            report=runner.main(a,backend=backend,sampler_factory=Mock(return_value=sampler))
            self.assertEqual(len(jobs),5);self.assertEqual(report['configuration']['total_shots'],112)
            for profile in report['statistics']['profiles'].values():np.testing.assert_allclose(profile['populations'],.5,atol=1e-14)
            self.assertEqual(report['verification']['compiled_batches_not_available'],[])
            directory=Path(report['run_directory']);raw={p.name:p.read_bytes() for p in directory.iterdir() if p.is_file()}
            forbidden=Mock(side_effect=AssertionError('no submission'))
            resumed=runner.parse_arguments(['--resume',str(directory/'manifest.json'),'--bootstrap','100',
                '--analysis-output-directory',str(Path(folder)/'resumed')])
            runner.main(resumed,sampler_factory=forbidden);forbidden.assert_not_called()
            for name,value in raw.items():
                if name not in ('results_bundle.zip','hardware_results.json'):self.assertEqual((directory/name).read_bytes(),value)
            for p in directory.glob('*.qpy'):p.unlink()
            portable=runner.analyze(directory,replicates=100)
            self.assertEqual(len(portable['verification']['compiled_batches_not_available']),5)

    def test_unknown_submission_is_never_retried_and_source_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()), \
             patch.object(hardware,'make_service',side_effect=AssertionError('no account')):
            a,backend=self.options(folder,'--execute');sampler=Mock();sampler.run.side_effect=RuntimeError('lost submission response')
            with self.assertRaisesRegex(RuntimeError,'lost submission'):
                runner.main(a,backend=backend,sampler_factory=Mock(return_value=sampler))
            manifest=next((Path(folder)/'mitigation').glob('*/manifest.json'))
            forbidden=Mock(side_effect=AssertionError('no retry'))
            a=runner.parse_arguments(['--resume',str(manifest)])
            with self.assertRaisesRegex(ValueError,'no recorded job ID'):runner.main(a,sampler_factory=forbidden)
            forbidden.assert_not_called()
            (manifest.parent/'source.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):runner.main(a,sampler_factory=forbidden)

    def test_known_job_recovery_preserves_initial_snapshot_and_avoids_duplicate_jobs(self):
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()), \
             patch.object(hardware,'make_service',side_effect=AssertionError('no account')):
            a,backend=self.options(folder,'--execute');sampler,jobs=self.sampler();original=sampler.run.side_effect
            def interrupted(pubs,shots):
                job=original(pubs,shots);job.result.side_effect=RuntimeError('connection interrupted');return job
            sampler.run.side_effect=interrupted
            with self.assertRaisesRegex(RuntimeError,'connection interrupted'):
                runner.main(a,backend=backend,sampler_factory=Mock(return_value=sampler))
            manifest=next((Path(folder)/'mitigation').glob('*/manifest.json'))
            snapshot=(manifest.parent/'backend_snapshot.json').read_bytes()
            jobs[0].result.side_effect=None;service=Mock();service.job.return_value=jobs[0];sampler.run.side_effect=original
            a=runner.parse_arguments(['--resume',str(manifest),'--bootstrap','100','--analysis-output-directory',str(Path(folder)/'resumed')])
            report=runner.main(a,backend=backend,service=service,sampler_factory=Mock(return_value=sampler))
            self.assertEqual(len(jobs),5);service.job.assert_called_once_with('local_mitigation_0')
            self.assertEqual(len(report['job_ids']),5)
            self.assertEqual(snapshot,(manifest.parent/'backend_snapshot.json').read_bytes())
            self.assertEqual(len(list((manifest.parent/'resume_calibration').glob('*/backend_snapshot.json'))),1)

    def test_modified_qpy_is_rejected_and_reference_values_do_not_determine_correction(self):
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()), \
             patch.object(hardware,'make_service',side_effect=AssertionError('no account')):
            a,backend=self.options(folder,'--compile-only');manifest=runner.main(a,backend=backend)
            directory=next((Path(folder)/'mitigation').iterdir());source,_=runner.read_source(directory,manifest)
            batch=manifest['batches'][0];path=directory/batch['isa_file'];path.write_bytes(path.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'QPY hash mismatch'):runner.compiled_batch(directory,manifest,batch,source)
        # Synthetic response has a known ideal intercept; neither reporting
        # reference enters calibration or the extrapolated population estimator.
        values=np.full((3,4,1),.4);cov=np.full((3,4,1,1),.001)
        controls={arm:{'all_zero':[{'0':100}],'all_one':[{'1':100}],
            'initial_mixture':[{'0':50,'1':50}]} for arm in ('raw','twirled')}
        a=mit.estimates(values,cov,controls,[1,1.25,1.5],np.full((3,1),.4),np.array([.5]),replicates=100)
        b=mit.estimates(values,cov,controls,[1,1.25,1.5],np.full((3,1),.9),np.array([.1]),replicates=100)
        for name in a['profiles']:np.testing.assert_array_equal(a['profiles'][name]['populations'],b['profiles'][name]['populations'])


if __name__=='__main__':unittest.main()
