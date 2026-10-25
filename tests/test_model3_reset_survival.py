"""Physical timing controls with local targets and mocked Runtime only."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

from qiskit import qpy
from Model3.Experiment5 import run_reset_survival as runner, run_hardware as hardware
from tests import test_model3_experiment5_hardware as fixtures
from tests import test_model3_preparation_diagnostic as prep


class ResetSurvivalTests(unittest.TestCase):
    def setup(self, folder, *extra):
        reference,backend=prep.PreparationTests().setup(folder)
        options=runner.parse_arguments(["--reference-run",str(reference.reference_run),"--steps","2","4",
            "--shots","8","--output-directory",str(Path(folder)/"runs"),
            "--analysis-output-directory",str(Path(folder)/"analysis"),*extra])
        return options,backend

    def sampler(self):
        sampler=Mock();job=Mock();job.job_id.return_value="offline_survival_job"
        job.result.side_effect=lambda:[fixtures.pub({"00":4,"01":4}) for _ in sampler.run.call_args.args[0]]
        job.metrics.return_value={};sampler.run.return_value=job
        return sampler,job

    def test_default_plan_budget_and_invalid_settings(self):
        options=runner.parse_arguments([])
        backend=fixtures.HardwareRunnerTests().backend()
        cfg=dict(steps=options.steps,idle_scales=options.idle_scales,
                 timing_model=runner.timing_model(backend,[0,1,2,3,4]))
        cases=runner.plan(cfg)
        self.assertEqual(len(cases),64)
        self.assertEqual(len(cases)*options.shots,131072)
        self.assertEqual(len([c for c in cases if c['arm']=='baseline']),4)
        for args in [["--steps","0"],["--steps","1","1"],["--idle-scales",".5"],
                     ["--idle-scales","1","nan"],["--idle-scales","1","1"],
                     ["--shots","1"],["--order-seed","-1"]]:
            with self.subTest(args=args),self.assertRaises(ValueError):runner.parse_arguments(args)

    def test_all_native_controls_have_exact_modeled_waits_and_no_exchange(self):
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()), \
             patch.object(hardware,"make_service",side_effect=AssertionError("no account")):
            options,backend=self.setup(folder,"--compile-only")
            manifest=runner.main(options,backend=backend,sampler_factory=Mock(side_effect=AssertionError("no submission")))
            cfg=manifest['configuration'];self.assertEqual(cfg['total_circuits'],44)
            file=next((Path(folder)/"runs").glob('*/isa_circuits.qpy'))
            with file.open('rb') as f:circuits=qpy.load(f)
            for case,circuit in zip(manifest['circuits'],circuits):
                pair=cfg['timing_model']['pairs'][case['side']]
                self.assertEqual(case['schedule']['summed_pre_measure_dt'],pair['prepare_dt']+case['modeled_wait_dt'])
                self.assertEqual(case['final_logical_to_physical'],[pair['system'],pair['bath']])
                ops=circuit.count_ops();self.assertEqual(ops['measure'],2)
                self.assertFalse(ops.get('cz',0))
                resets=case['steps'] if case['arm'].startswith('reset_') else 0
                self.assertEqual(ops.get('reset',0),resets)
                self.assertEqual(ops.get('x',0),case['initial_system']+(case['steps'] if case['arm']=='reset_one' else 0))
                # Explicit physical mapping: classical bit 0 is system, bit 1 bath.
                measured=[(circuit.find_bit(n.qubits[0]).index,circuit.find_bit(n.clbits[0]).index)
                          for n in circuit.data if n.operation.name=='measure']
                self.assertEqual(measured,[(pair['system'],0),(pair['bath'],1)])
                if case['arm']=='idle' and case['idle_scale']==1:
                    self.assertEqual(case['modeled_wait_dt'],case['steps']*pair['cycle_dt'])
            self.assertTrue((file.parent/'results_bundle.zip').is_file())
            self.assertFalse((file.parent/'hardware_results.json').exists())

    def test_padding_keeps_target_alignment_and_static_model_equality(self):
        backend=fixtures.HardwareRunnerTests().backend()
        backend.target.pulse_alignment=16;backend.target.acquire_alignment=16
        # Small aligned-duration target properties, deliberately requiring padding.
        from qiskit.transpiler import InstructionProperties
        for q in [0,1,3,4]:
            backend.target.update_instruction_properties('x',(q,),InstructionProperties(duration=160*backend.dt,error=.001))
            backend.target.update_instruction_properties('reset',(q,),InstructionProperties(duration=1000*backend.dt))
        cfg=dict(num_qubits=backend.num_qubits,steps=[2],idle_scales=[.5,1.,2.],
                 timing_model=runner.timing_model(backend,[0,1,2,3,4]))
        for case in runner.plan(cfg):
            circuit=runner.build(cfg,case)
            summary=runner.schedule_summary(circuit,backend.target)
            pair=cfg['timing_model']['pairs'][case['side']]
            self.assertEqual(case['cycle_dt']%16,0)
            self.assertEqual(summary['summed_pre_measure_dt'],pair['prepare_dt']+case['modeled_wait_dt'])

    def test_collect_analyze_preserve_raw_and_reject_modified_circuits(self):
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()), \
             patch.object(hardware,'make_service',side_effect=AssertionError('no account')):
            options,backend=self.setup(folder);sampler,_=self.sampler()
            report=runner.main(options,backend=backend,sampler_factory=Mock(return_value=sampler))
            self.assertEqual(len(report['cases']),44)
            self.assertEqual(len(report['comparisons']),16)
            self.assertTrue(all(r['system_population']==.5 and r['bath_population']==0 for r in report['cases']))
            self.assertTrue(all(r['reset_minus_idle']==0 for r in report['comparisons']))
            directory=Path(report['run_directory']);raw={p.name:p.read_bytes() for p in directory.iterdir() if p.is_file()}
            with zipfile.ZipFile(directory/'results_bundle.zip') as z:
                self.assertIn('isa_circuits.qpy',z.namelist());self.assertIn('analysis/reset_survival.png',z.namelist())
            offline=runner.parse_arguments(['--analyze-only',str(directory),'--analysis-output-directory',str(Path(folder)/'offline')])
            runner.main(offline,sampler_factory=Mock(side_effect=AssertionError('no submission')))
            for name,data in raw.items():self.assertEqual((directory/name).read_bytes(),data)
            manifest=json.loads((directory/'manifest.json').read_bytes())
            manifest['circuits'][0]['schedule']['summed_pre_measure_dt']+=1
            with self.assertRaisesRegex(ValueError,'timing mismatch'):runner.verify_plan(manifest)
            with (directory/'isa_circuits.qpy').open('rb') as f:pubs=qpy.load(f)
            pubs[0].x(0)
            original=json.loads(raw['manifest.json'])
            with self.assertRaisesRegex(ValueError,'native control plan'):runner.verify_circuits(pubs,original)
            # The archive is sufficient when large QPY is not separately pulled.
            (directory/'isa_circuits.qpy').unlink()
            runner.analyze(directory)

    def test_interrupted_job_and_completed_resume_never_resubmit(self):
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()), \
             patch.object(hardware,'make_service',side_effect=AssertionError('no account')):
            options,backend=self.setup(folder);sampler,job=self.sampler()
            job.result.side_effect=RuntimeError('interrupted')
            with self.assertRaisesRegex(RuntimeError,'interrupted'):
                runner.main(options,backend=backend,sampler_factory=Mock(return_value=sampler))
            path=next((Path(folder)/'runs').glob('*/manifest.json'))
            job.result.side_effect=None;job.result.return_value=[fixtures.pub({'00':8}) for _ in range(44)]
            service=Mock();service.job.return_value=job
            options=runner.parse_arguments(['--resume',str(path),'--analysis-output-directory',str(Path(folder)/'analysis')])
            forbidden=Mock(side_effect=AssertionError('no resubmission'))
            report=runner.main(options,backend=backend,service=service,sampler_factory=forbidden)
            self.assertEqual(report['job_ids'],['offline_survival_job'])
            service.job.assert_called_once_with('offline_survival_job');forbidden.assert_not_called()
            # A completed resume is fully offline and must not need a backend.
            runner.main(options,sampler_factory=forbidden)


if __name__=='__main__':unittest.main()
