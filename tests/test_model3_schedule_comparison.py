"""Paired hardware workflow checked only with local targets/mocked Runtime."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
import zipfile

import numpy as np
from qiskit import qpy
from library import reservoir_scaling as physics
from Model3.Experiment5 import run_schedule_comparison as runner,run_hardware as hardware
from tests import test_model3_experiment5_hardware as fixtures


class ComparisonTests(unittest.TestCase):
    def setup(self,folder,*extra):
        backend=fixtures.HardwareRunnerTests().backend();base=fixtures.source();candidate=copy.deepcopy(base)
        candidate['schedule']['step_dts']=candidate['schedule']['step_dts'][:1]
        candidate['schedule']['jump_choices']=[x[:1] for x in candidate['schedule']['jump_choices']]
        candidate['times']=candidate['times'][:2]
        model=physics.Model(3,.8/1.7,.9,.1,.5)
        for source in [base,candidate]:
            C=np.repeat((np.eye(3)/2)[None,None],2,axis=1)
            for k,dt in enumerate(source['schedule']['step_dts']):
                choices=np.array(source['schedule']['jump_choices'])[:,k]
                C=physics.sampled_step(C,choices,physics.channel(model,1.7*dt,'rate-weighted','hamiltonian-first','even-odd'))
            source['final_covariances_real_gauge']=C.tolist()
        archive=Path(folder)/'inputs.zip';meta=dict(backend=backend.name,physical_path=[0,1,2,3,4],sources={})
        raw={label:json.dumps(source).encode() for label,source in [('baseline',base),('candidate',candidate)]}
        for label,value in raw.items():meta['sources'][label]=dict(file=label+'.json',sha256=hashlib.sha256(value).hexdigest())
        with zipfile.ZipFile(archive,'w') as z:
            z.writestr('input_manifest.json',json.dumps(meta))
            for label,value in raw.items():z.writestr(label+'.json',value)
        options=runner.parse_arguments(['--input-bundle',str(archive),'--backend',backend.name,'--trajectory-count','2',
            '--shots','8','--control-shots','16','--batch-size','3','--output-directory',str(Path(folder)/'runs'),
            '--analysis-output-directory',str(Path(folder)/'analysis'),*extra])
        return options,backend

    def sampler(self):
        sampler=Mock();jobs=[]
        def submit(pubs,shots):
            job=Mock();job.job_id.return_value=f'offline_paired_{len(jobs)}'
            job.result.return_value=[fixtures.pub({'000':spec[2]//2,'111':spec[2]//2}) for spec in pubs]
            job.metrics.return_value={};jobs.append(job);return job
        sampler.run.side_effect=submit
        return sampler,jobs

    def test_default_budget_and_paired_batches(self):
        options=runner.parse_arguments([]);sources,_,meta,_=runner.load_inputs(options.input_bundle)
        cfg=dict(trajectory_count=options.trajectory_count,shots=options.shots,control_shots=options.control_shots,
                 batch_size=options.batch_size,order_seed=options.order_seed)
        records,batches=runner.plan(cfg,sources)
        self.assertEqual(len(records),2051);self.assertEqual(sum(r['shots'] for r in records),22528)
        self.assertEqual(len(batches),5);self.assertEqual(meta['physical_path'],[39,49,48,58,57,56,66,76,86,85,95,105])
        for b in batches:
            selected=records[b['start']:b['stop']];self.assertLessEqual(len(selected),512)
            trajectories={r['trajectory'] for r in selected if r['kind']=='trajectory'}
            for index in trajectories:self.assertEqual(sum(r.get('trajectory')==index for r in selected),2)
        for args in [['--shots','1'],['--batch-size','1'],['--trajectory-count','1'],['--order-seed','-1']]:
            with self.subTest(args=args),self.assertRaises(ValueError):runner.parse_arguments(args)

    def test_compile_only_saves_native_chunks_without_submission(self):
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()), \
             patch.object(hardware,'make_service',side_effect=AssertionError('no account')):
            options,backend=self.setup(folder,'--compile-only');forbidden=Mock(side_effect=AssertionError('no submission'))
            manifest=runner.main(options,backend=backend,sampler_factory=forbidden)
            self.assertEqual(manifest['status'],'compiled');self.assertEqual(len(manifest['batches']),3)
            directory=next((Path(folder)/'runs').iterdir())
            for batch in manifest['batches']:
                with (directory/batch['isa_file']).open('rb') as f:pubs=qpy.load(f)
                for pub,record in zip(pubs,manifest['circuits'][batch['start']:batch['stop']]):
                    if record['kind']=='trajectory':
                        steps=2 if record['schedule']=='baseline' else 1
                        self.assertEqual(pub.count_ops()['cz'],6*steps)
                        self.assertEqual(pub.count_ops()['reset'],steps)
                    else:self.assertFalse(pub.count_ops().get('cz',0))
                    self.assertEqual(record['final_logical_to_physical'],[1,2,3,0,4])
            forbidden.assert_not_called()

    def test_collection_analysis_archive_and_offline_resume_preserve_files(self):
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()), \
             patch.object(hardware,'make_service',side_effect=AssertionError('no account')):
            options,backend=self.setup(folder);sampler,jobs=self.sampler()
            report=runner.main(options,backend=backend,sampler_factory=Mock(return_value=sampler))
            self.assertEqual(len(jobs),3);self.assertEqual(report['configuration']['total_shots'],80)
            for value in report['schedules'].values():np.testing.assert_allclose(value['hardware_populations'],.5)
            expected=report['schedules']['baseline']['ideal_same_paths']-report['schedules']['candidate']['ideal_same_paths']
            np.testing.assert_allclose(report['paired_residual_difference'],expected)
            directory=Path(report['run_directory']);raw={p.name:p.read_bytes() for p in directory.iterdir() if p.is_file()}
            with zipfile.ZipFile(directory/'results_bundle.zip') as z:
                self.assertIn('analysis/populations.png',z.namelist());self.assertFalse(any(n.endswith('.qpy') for n in z.namelist()))
            forbidden=Mock(side_effect=AssertionError('no resubmission'))
            offline=runner.parse_arguments(['--analyze-only',str(directory),'--analysis-output-directory',str(Path(folder)/'offline')])
            runner.main(offline,sampler_factory=forbidden)
            for name,value in raw.items():self.assertEqual((directory/name).read_bytes(),value)
            resumed=runner.parse_arguments(['--resume',str(directory/'manifest.json'),'--analysis-output-directory',str(Path(folder)/'resumed')])
            runner.main(resumed,sampler_factory=forbidden);forbidden.assert_not_called()
            for p in directory.glob('*.qpy'):p.unlink()
            portable=runner.analyze(directory)
            self.assertEqual(len(portable['verification']['compiled_batches_not_available']),3)

    def test_interrupted_collection_recovers_job_and_submits_only_remaining_batches(self):
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()), \
             patch.object(hardware,'make_service',side_effect=AssertionError('no account')):
            options,backend=self.setup(folder);sampler,jobs=self.sampler()
            original=sampler.run.side_effect
            def interrupted(pubs,shots):
                job=original(pubs,shots);job.result.side_effect=RuntimeError('interrupted');return job
            sampler.run.side_effect=interrupted
            with self.assertRaisesRegex(RuntimeError,'interrupted'):runner.main(options,backend=backend,sampler_factory=Mock(return_value=sampler))
            manifest=next((Path(folder)/'runs').glob('*/manifest.json'))
            jobs[0].result.side_effect=None
            service=Mock();service.job.return_value=jobs[0]
            sampler.run.side_effect=original
            resume=runner.parse_arguments(['--resume',str(manifest),'--analysis-output-directory',str(Path(folder)/'analysis')])
            report=runner.main(resume,backend=backend,service=service,sampler_factory=Mock(return_value=sampler))
            service.job.assert_called_once_with('offline_paired_0');self.assertEqual(len(jobs),3)
            self.assertEqual(len(report['job_ids']),3)

    def test_unknown_submission_id_is_never_automatically_retried(self):
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()), \
             patch.object(hardware,'make_service',side_effect=AssertionError('no account')):
            options,backend=self.setup(folder);sampler=Mock();sampler.run.side_effect=RuntimeError('lost response')
            with self.assertRaisesRegex(RuntimeError,'lost response'):
                runner.main(options,backend=backend,sampler_factory=Mock(return_value=sampler))
            manifest=next((Path(folder)/'runs').glob('*/manifest.json'));forbidden=Mock(side_effect=AssertionError('no retry'))
            resume=runner.parse_arguments(['--resume',str(manifest)])
            with self.assertRaisesRegex(ValueError,'before job ID'):
                runner.main(resume,backend=backend,sampler_factory=forbidden)
            forbidden.assert_not_called()

    def test_changed_source_or_qpy_rejected_before_any_job(self):
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()), \
             patch.object(hardware,'make_service',side_effect=AssertionError('no account')):
            options,backend=self.setup(folder,'--compile-only');manifest=runner.main(options,backend=backend)
            directory=next((Path(folder)/'runs').iterdir());sources,_=runner.read_sources(directory,manifest['configuration'])
            batch=manifest['batches'][0];file=directory/batch['isa_file'];file.write_bytes(file.read_bytes()+b'bad')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):runner.compiled_batch(directory,manifest,batch,sources)
            source=directory/'candidate.json';source.write_text('{}')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):runner.read_sources(directory,manifest['configuration'])


if __name__=='__main__':unittest.main()
