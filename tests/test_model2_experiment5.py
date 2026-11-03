"""Model2 hardware physics, fixed-path provenance, compilation and job recovery."""
import contextlib
import copy
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

import numpy as np
from qiskit import qpy, QuantumCircuit
from qiskit.qpy import common as qpy_common
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import DensityMatrix, Kraus, partial_trace

from library import reservoir_scaling as physics
from Model2.Experiment5 import input_data as inputs
from Model2.Experiment5 import run_hardware as experiment
from Model2.Experiment5 import analyze
from Model2.Experiment5 import qpy_storage


def nonselective_density(circuit):
    density = DensityMatrix.from_label("0"*circuit.num_qubits)
    dephase = Kraus([np.diag([1,0]), np.diag([0,1])])
    for instruction in circuit.data:
        args = [circuit.find_bit(q).index for q in instruction.qubits]
        if instruction.operation.name == "measure":
            density = density.evolve(dephase,args)
        elif instruction.operation.name == "reset":
            density = density.reset(args)
        else:
            density = density.evolve(instruction.operation,args)
    return density


def covariance(density,n):
    operators=[]
    for j in range(n):
        value=np.ones((1,1))
        for k in reversed(range(n)):
            value=np.kron(value, np.diag([1,-1]) if k<j else np.array([[0,1],[0,0]]) if k==j else np.eye(2))
        operators.append(value)
    return np.array([[np.trace(b.conj().T@a@density.data) for b in operators] for a in operators])


class Model2HardwareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory()
        cls.root=Path(cls.temporary.name)
        cls.backend=GenericBackendV2(12,basis_gates=["cz","rz","sx","x"],
            coupling_map=[[i,i+1] for i in range(11)]+[[i+1,i] for i in range(11)],seed=17)
        args=["--backend",cls.backend.name,"--trajectory-count","2","--shots","4","--control-shots","8",
              "--batch-size","9","--physical-path",*map(str,range(12)),"--min-t1-us","1","--min-t2-us","1",
              "--max-cz-error",".1","--max-readout-error",".1","--output-directory",str(cls.root),"--compile-only"]
        with contextlib.redirect_stdout(io.StringIO()), patch.object(experiment.runtime,"make_service",side_effect=AssertionError("no account")):
            experiment.main(experiment.parse_arguments(args),backend=cls.backend)
        cls.template=next(cls.root.glob("cosine44_*"))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def saved(self,folder):
        destination=Path(folder)/"run"
        shutil.copytree(self.template,destination)
        return destination

    def legacy_crash(self, folder):
        destination = self.saved(folder)
        manifest = json.loads((destination/"manifest.json").read_text())
        manifest["status"] = "planned"
        config = manifest["configuration"]
        config.update(isa_sha256=None, physical_path=None)
        for key in ("isa_storage", "isa_index_sha256", "versions"):
            config.pop(key, None)
        for batch in manifest["batches"]:
            for key in ("isa_file", "isa_sha256", "isa_byte_range"):
                batch.pop(key, None)
        for row in manifest["circuits"]:
            row.pop("metrics", None)
            row.pop("final_logical_to_physical", None)
        manifest["plan_sha256"] = experiment.plan_hash(manifest)
        (destination/"manifest.json").write_text(json.dumps(manifest))
        return destination

    def no_large_read_bytes(self, path):
        if path.suffix == ".qpy":
            raise AssertionError("QPY files must be hashed in bounded chunks")
        with path.open("rb") as handle:
            return handle.read()

    def compile_options(self, folder):
        return experiment.parse_arguments(["--compile-only", "--backend", self.backend.name,
            "--trajectory-count", "2", "--shots", "4", "--control-shots", "8", "--batch-size", "6",
            "--physical-path", *map(str, range(12)), "--min-t1-us", "1", "--min-t2-us", "1",
            "--max-cz-error", ".1", "--max-readout-error", ".1", "--output-directory", str(folder)])

    def test_legacy_crash_recovery_is_offline_unchanged_and_does_not_recompile(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            folder = self.legacy_crash(temporary)
            old_hash = qpy_storage.file_sha256(folder/"isa_circuits.qpy")
            with patch.object(experiment.runtime, "make_service", side_effect=AssertionError("offline recovery")), \
                 patch.object(experiment, "transpile", side_effect=AssertionError("reuse existing compilation")), \
                 patch.object(Path, "read_bytes", self.no_large_read_bytes):
                result = experiment.main(experiment.parse_arguments(["--resume", str(folder/"manifest.json"), "--recover-compiled"]))
                self.assertEqual(result["status"], "compiled")
                self.assertEqual(result["configuration"]["physical_path"], list(range(12)))
                self.assertEqual(result["configuration"]["isa_sha256"], old_hash)
                self.assertEqual(result["configuration"]["isa_storage"], "legacy-indexed-qpy-v1")
                self.assertFalse(result["compilation_recovery"]["scope"] == "")
                self.assertTrue(all(b["job_id"] is None for b in result["batches"]))
                folder, saved, _ = experiment.read_saved(folder/"manifest.json")
                store = experiment.Publications(saved, folder, self.backend)
                self.assertEqual(len(store[0:9]), 9)
                sampler, job, _ = self.sampler(folder)
                executed = experiment.main(experiment.parse_arguments(["--resume", str(folder/"manifest.json"), "--execute"]),
                    backend=self.backend, service=Mock(), sampler_factory=Mock(return_value=sampler))
                self.assertEqual(executed["job_ids"], ["model2_test_job"])
            self.assertEqual(qpy_storage.file_sha256(folder/"isa_circuits.qpy"), old_hash)

    def test_incomplete_legacy_archive_leaves_manifest_and_archive_untouched(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            folder = self.legacy_crash(temporary)
            path = folder/"isa_circuits.qpy"
            with path.open("r+b") as handle:
                handle.truncate(path.stat().st_size-20)
            before = (folder/"manifest.json").read_bytes()
            damaged_hash = qpy_storage.file_sha256(path)
            with patch.object(experiment.runtime, "make_service", side_effect=AssertionError("offline recovery")):
                with self.assertRaises(Exception):
                    experiment.main(experiment.parse_arguments(["--resume", str(folder/"manifest.json"), "--recover-compiled"]))
            self.assertEqual(before, (folder/"manifest.json").read_bytes())
            self.assertEqual(damaged_hash, qpy_storage.file_sha256(path))

    def test_qpy_batch_windows_preserve_payload_across_format_versions(self):
        programs = []
        for index in range(5):
            circuit = QuantumCircuit(3, 3, name=f"stored_{index}")
            circuit.h(index % 3)
            circuit.rz(.123*(index+1), 0)
            circuit.cz(0, 1)
            circuit.measure(0, 0)
            circuit.reset(0)
            circuit.metadata = {"identity": index}
            programs.append(circuit)
        ranges = [{"start": 0, "stop": 2}, {"start": 2, "stop": 4}, {"start": 4, "stop": 5}]
        for version in sorted({13, 15, 16, qpy_common.QPY_VERSION}):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary)/"original.qpy"
                with path.open("wb") as handle:
                    qpy.dump(programs, handle, version=version)
                archive = qpy_storage.Archive(path, len(programs))
                indexed = []
                for batch, (loaded, byte_range) in zip(ranges, archive.iter_batches(ranges)):
                    self.assertEqual(loaded, programs[batch["start"]:batch["stop"]])
                    self.assertEqual([p.metadata for p in loaded], [p.metadata for p in programs[batch["start"]:batch["stop"]]])
                    indexed.append(dict(batch, isa_byte_range=byte_range))
                self.assertEqual(archive.load_batch(indexed[1]), programs[2:4])
                with path.open("ab") as handle:
                    handle.write(b"unexpected trailing bytes")
                with self.assertRaisesRegex(ValueError, "trailing|missing"):
                    list(qpy_storage.Archive(path, len(programs)).iter_batches(ranges))

    def test_large_plans_store_and_load_one_job_at_a_time(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(experiment.runtime, "make_service", side_effect=AssertionError("mock backend only")), \
             patch.object(Path, "read_bytes", self.no_large_read_bytes):
            manifest = experiment.main(self.compile_options(temporary), backend=self.backend)
            folder = next(Path(temporary).glob("cosine44_*"))
            self.assertEqual(manifest["configuration"]["isa_storage"], "batch-qpy-v1")
            self.assertEqual(len(manifest["batches"]), 2)
            self.assertFalse((folder/"isa_circuits.qpy").exists())
            _, saved, _ = experiment.read_saved(folder/"manifest.json")
            store = experiment.Publications(saved, folder, self.backend)
            self.assertEqual(len(store), 12)
            for batch in saved["batches"]:
                self.assertEqual(len(store[batch["start"]:batch["stop"]]), 6)
            with self.assertRaisesRegex(ValueError, "complete saved job"):
                store[0:12]
            experiment.bundle(folder)
            with zipfile.ZipFile(folder/"results_bundle.zip") as archive:
                self.assertTrue(all(b["isa_file"] in archive.namelist() for b in saved["batches"]))
            sampler = Mock()
            submitted = []
            def submit(items, shots):
                current = json.loads((folder/"manifest.json").read_text())
                self.assertEqual(len(items), 6)
                pubs = []
                for circuit, _, requested in items:
                    record = current["circuits"][circuit.metadata["pub_index"]]
                    self.assertEqual(requested, record["shots"])
                    counts = ({"1"*10:requested} if record.get("control") == "all_one" else
                              {"0"*10:requested//2, "1"*10:requested-requested//2} if record.get("control") == "half_mixed" else
                              {"0"*10:requested})
                    pubs.append(SimpleNamespace(data=SimpleNamespace(outcome=SimpleNamespace(get_counts=lambda counts=counts:counts))))
                job = Mock()
                job.job_id.return_value = f"bounded_job_{len(submitted)}"
                job.result.return_value = pubs
                job.metrics.return_value = {}
                submitted.append(job)
                return job
            sampler.run.side_effect = submit
            result = experiment.main(experiment.parse_arguments(["--resume", str(folder/"manifest.json"), "--execute"]),
                backend=self.backend, service=Mock(), sampler_factory=Mock(return_value=sampler))
            self.assertEqual(result["job_ids"], ["bounded_job_0", "bounded_job_1"])
            self.assertEqual(sampler.run.call_count, 2)

    def test_interrupted_batch_compilation_resumes_without_recompiling_saved_paths(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            dump = qpy.dump
            calls = 0
            def interrupted(programs, handle):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("interrupted local write")
                return dump(programs, handle)
            with patch.object(experiment.qpy, "dump", side_effect=interrupted), self.assertRaisesRegex(RuntimeError, "local write"):
                experiment.main(self.compile_options(temporary), backend=self.backend)
            folder = next(Path(temporary).glob("cosine44_*"))
            _, saved, _ = experiment.read_saved(folder/"manifest.json")
            completed = saved["batches"][0]
            original_hash = qpy_storage.file_sha256(folder/completed["isa_file"])
            old_path = next(r["trajectory"] for r in saved["circuits"][completed["start"]:completed["stop"]] if r["kind"] == "evolution")
            transpile = experiment.transpile
            def forbid_recompile(body, **kwargs):
                if body.metadata["steps"] == 44 and body.metadata["trajectory"] == old_path:
                    raise AssertionError("completed path recompiled")
                return transpile(body, **kwargs)
            with patch.object(experiment, "transpile", side_effect=forbid_recompile):
                done = experiment.main(experiment.parse_arguments(["--resume", str(folder/"manifest.json"), "--compile-only"]), backend=self.backend)
            self.assertEqual(done["status"], "compiled")
            self.assertEqual(qpy_storage.file_sha256(folder/completed["isa_file"]), original_hash)

    def test_streaming_loader_preserves_156_qubit_native_layout_and_gates(self):
        from qiskit_ibm_runtime.fake_provider import FakeKingston
        backend = FakeKingston()
        physical = [95, 94, 93, 79, 73, 74, 75, 59, 55, 54, 53, 52]
        layout = physical[1:-1]+[physical[0], physical[-1]]
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(experiment.runtime, "make_service", side_effect=AssertionError("offline target")):
            args = ["--compile-only", "--backend", backend.name, "--trajectory-count", "2", "--shots", "4",
                    "--batch-size", "9", "--physical-path", *map(str, physical), "--min-t1-us", "1",
                    "--min-t2-us", "1", "--max-cz-error", ".1", "--max-readout-error", ".1",
                    "--output-directory", temporary]
            manifest = experiment.main(experiment.parse_arguments(args), backend=backend)
            folder = next(Path(temporary).glob("cosine44_*"))
            loaded = experiment.Publications(manifest, folder, backend)[0:9]
            with (folder/"isa_circuits.qpy").open("rb") as handle:
                reference = qpy.load(handle)
            self.assertEqual(loaded, reference)
            for circuit in loaded:
                self.assertEqual(circuit.num_qubits, 156)
                self.assertEqual(circuit.layout.final_index_layout(filter_ancillas=True), layout)

    def sampler(self,folder,job_id="model2_test_job"):
        sampler=Mock();job=Mock();job.job_id.return_value=job_id;job.metrics.return_value={}
        def result():
            saved=json.loads((folder/"manifest.json").read_text())
            self.assertEqual(saved["batches"][0]["job_id"],job_id)
            pubs=[]
            for item in sampler.run.call_args.args[0]:
                circuit=item[0] if isinstance(item,tuple) else item
                row=saved["circuits"][circuit.metadata["pub_index"]]
                shots=row["shots"]
                counts=({"1"*10:shots} if row.get("control")=="all_one" else
                        {"0"*10:shots//2,"1"*10:shots-shots//2} if row.get("control")=="half_mixed" else
                        {"0"*10:shots})
                pubs.append(SimpleNamespace(data=SimpleNamespace(outcome=SimpleNamespace(get_counts=lambda counts=counts:counts),
                    preparation=SimpleNamespace(get_counts=Mock(side_effect=AssertionError("ignore preparation bits"))))))
            return pubs
        job.result.side_effect=result
        sampler.run.return_value=job
        return sampler,job,result

    def test_frozen_spec_and_changed_parameters_are_rejected(self):
        spec=inputs.load_spec()
        self.assertEqual(spec["physics"]["exchange_prefactor"],.25)
        self.assertEqual(spec["schedule"]["steps"],44)
        with tempfile.TemporaryDirectory() as folder:
            altered=copy.deepcopy(spec);altered["schedule"]["power"]+=.01
            path=Path(folder)/"altered.json";path.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError,"frozen"):
                inputs.load_spec(path)

    def test_paths_have_prefixes_and_match_original_full_replay_hash(self):
        spec=inputs.load_spec();probabilities=[.125,.375,.125,.375]
        short=inputs.choices(29,4,44,probabilities)
        np.testing.assert_array_equal(short,inputs.choices(29,17,44,probabilities)[:4])
        full=inputs.make_source(spec,16384,29)
        self.assertEqual(full["provenance"]["jump_choices_sha256"],spec["validation"]["replays"][0]["choices_sha256"])
        self.assertTrue(full["finite_path_assessment"]["empirical_classical_checks_passed"])

    def test_default_plan_is_offline_and_does_not_inherit_full_sampling_validation(self):
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()), \
             patch.object(experiment.runtime,"make_service",side_effect=AssertionError("no account")), \
             patch.object(experiment,"transpile",side_effect=AssertionError("no transpilation")):
            manifest=experiment.main(experiment.parse_arguments(["--output-directory",folder]))
            self.assertEqual(manifest["configuration"]["total_jobs"],7)
            self.assertEqual(manifest["configuration"]["total_circuits"],3093)
            self.assertEqual(manifest["configuration"]["total_shots"],67584)
            path=next(Path(folder).glob("*/manifest.json"))
            _,_,source=experiment.read_saved(path)
            self.assertFalse(source["finite_path_assessment"]["empirical_classical_checks_passed"])
            self.assertAlmostEqual(source["finite_path_assessment"]["ideal_subset_target_error"],.020136387318241245)
            for job in manifest["batches"]:
                rows=manifest["circuits"][job["start"]:job["stop"]]
                self.assertEqual({r["control"] for r in rows if r["kind"]=="control"},{"all_zero","all_one","half_mixed"})
                for path_id in {r["trajectory"] for r in rows if r["kind"]=="evolution"}:
                    self.assertEqual([r["basis"] for r in rows if r.get("trajectory")==path_id],["z","xy","yx"])

    def test_bad_cli_is_rejected_before_any_account(self):
        for args in (["--shots","0"],["--trajectory-count","1"],["--trajectory-count","16385"],
                     ["--batch-size","5"],["--bases","xy"],["--seed","-1"],
                     ["--resume","x.json","--shots","10"],["--physical-path","0","1"],["--recover-compiled"]):
            with self.subTest(args=args),self.assertRaises(ValueError):
                experiment.parse_arguments(args)

    def test_actual_large_angle_model2_circuit_matches_full_density(self):
        spec=inputs.load_spec();p={**spec["physics"],"n_qubits":3}
        dts=[spec["schedule"]["step_dts"][0],4.,3.9,spec["schedule"]["step_dts"][-1]]
        source={"configuration":{"physics":p,"trajectories":1,"order":"jump-first"},
            "preparation":{"populations":[.5]*3},
            "schedule":{"step_dts":dts,"jump_choices":[[0,1,2,3]],"jump_probabilities":[.125,.375,.125,.375]}}
        model=physics.Model(3,2,.25,.25,.25)
        C=np.eye(3)[None,None]/2
        phase=(-1j)**np.arange(3)
        for prefix,(dt,choice) in enumerate(zip(dts,[0,1,2,3]),1):
            branch=physics.channel(model,dt,"rate-weighted","jump-first","even-odd")
            self.assertLess(branch.cosines[0],0.)
            C=physics.sampled_step(C,[choice],branch)
            circuit=experiment.build_body(source,0,steps=prefix)
            reduced=partial_trace(nonselective_density(circuit),[3,4])
            np.testing.assert_allclose(covariance(reduced,3),phase[:,None]*C[0,0]*phase.conj()[None,:],atol=4e-14)
            self.assertEqual(circuit.count_ops()["reset"],prefix)

    def test_native_compilation_and_reload_preserve_frozen_protocol(self):
        folder,manifest,_=experiment.read_saved(self.template/"manifest.json")
        with (folder/"isa_circuits.qpy").open("rb") as stream:
            pubs=qpy.load(stream)
        experiment.validate_publications(manifest,pubs)
        self.assertEqual(manifest["configuration"]["physical_path"],list(range(12)))
        for row,circuit in zip(manifest["circuits"],pubs):
            experiment.circuits.assert_native(circuit,self.backend)
            if row["kind"]=="evolution":
                self.assertEqual(circuit.count_ops()["reset"],44)
                self.assertEqual(circuit.count_ops()["measure"],20)
                self.assertEqual(circuit.count_ops()["cz"],880)
        with self.assertRaisesRegex(ValueError,"identity"):
            experiment.validate_publications(manifest,pubs[::-1])

    def test_submission_controls_decoder_and_self_contained_bundle(self):
        with tempfile.TemporaryDirectory() as temporary,contextlib.redirect_stdout(io.StringIO()):
            folder=self.saved(temporary);sampler,job,_=self.sampler(folder)
            factory=Mock(return_value=sampler)
            result=experiment.main(experiment.parse_arguments(["--resume",str(folder/"manifest.json"),"--execute"]),
                backend=self.backend,service=Mock(),sampler_factory=factory)
            self.assertEqual(result["job_ids"],["model2_test_job"])
            np.testing.assert_array_equal(result["observables"]["populations"],np.zeros(10))
            np.testing.assert_array_equal(result["observables"]["excitation_flows"],np.zeros(9))
            self.assertAlmostEqual(result["local_observable_errors"]["physical_stationary"],.25)
            self.assertEqual(result["per_job_preparation_controls"][0]["maximum_held_out_residual"],0.)
            self.assertEqual(factory.call_args.kwargs["options"]["execution"],{"init_qubits":True})
            self.assertFalse(factory.call_args.kwargs["options"]["twirling"]["enable_gates"])
            with zipfile.ZipFile(folder/"results_bundle.zip") as archive:
                self.assertTrue({"manifest.json","source.json","spec.json","isa_circuits.qpy","hardware_results.json","Analysis.md"}<=set(archive.namelist()))
            with patch.object(experiment.runtime,"make_service",side_effect=AssertionError("collected data are offline")):
                experiment.main(experiment.parse_arguments(["--resume",str(folder/"manifest.json"),"--execute"]))

    def test_resume_retrieves_existing_job_without_resubmitting(self):
        with tempfile.TemporaryDirectory() as temporary,contextlib.redirect_stdout(io.StringIO()):
            folder=self.saved(temporary);sampler,job,success=self.sampler(folder)
            job.result.side_effect=RuntimeError("interrupted collection")
            opts=experiment.parse_arguments(["--resume",str(folder/"manifest.json"),"--execute"])
            with self.assertRaisesRegex(RuntimeError,"interrupted"):
                experiment.main(opts,backend=self.backend,service=Mock(),sampler_factory=Mock(return_value=sampler))
            job.result.side_effect=success
            service=Mock();service.job.return_value=job
            service.backend.side_effect=AssertionError("only collecting saved job")
            factory=Mock(side_effect=AssertionError("must not resubmit"))
            result=experiment.main(opts,service=service,sampler_factory=factory)
            service.job.assert_called_once_with("model2_test_job")
            factory.assert_not_called()
            self.assertEqual(result["job_ids"],["model2_test_job"])

    def test_ambiguous_submission_cannot_be_duplicated_on_resume(self):
        with tempfile.TemporaryDirectory() as temporary,contextlib.redirect_stdout(io.StringIO()):
            folder=self.saved(temporary);sampler=Mock();sampler.run.side_effect=TimeoutError("unknown submit outcome")
            opts=experiment.parse_arguments(["--resume",str(folder/"manifest.json"),"--execute"])
            with self.assertRaises(TimeoutError):
                experiment.main(opts,backend=self.backend,service=Mock(),sampler_factory=Mock(return_value=sampler))
            factory=Mock(side_effect=AssertionError("no duplicate submission"))
            with self.assertRaisesRegex(ValueError,"interrupted before its job ID"):
                experiment.main(opts,backend=self.backend,service=Mock(),sampler_factory=factory)
            factory.assert_not_called()

    def test_saved_source_native_circuits_and_immutable_settings_are_bound(self):
        for filename in ("source.json","isa_circuits.qpy","manifest.json"):
            with self.subTest(filename=filename),tempfile.TemporaryDirectory() as temporary:
                folder=self.saved(temporary)
                if filename=="manifest.json":
                    value=json.loads((folder/filename).read_text());value["configuration"]["shots"]=999
                    (folder/filename).write_text(json.dumps(value))
                else:
                    with (folder/filename).open("ab") as handle: handle.write(b" ")
                with self.assertRaisesRegex(ValueError,"hash|intact"):
                    experiment.read_saved(folder/"manifest.json")

    def test_current_sign_both_bond_parities_and_one_shot_features(self):
        source={"configuration":{"physics":{"n_qubits":3,"J":1.,"exchange_prefactor":.25}},
                "preparation":{"populations":[.5]*3}}
        manifest={"configuration":{"trajectories":2,"measurement_bases":["z","xy","yx"]},"circuits":[],"batches":[]}
        observations=[]
        for control,bitstring in [("all_zero","000"),("all_one","111"),("half_mixed",None)]:
            manifest["circuits"].append({"kind":"control","control":control,"basis":"z","batch":0,
                "shots":4,"pub_index":len(observations),"expected_populations":[.5]*3})
            observations.append({"outcome_counts":{bitstring:4} if bitstring else {"000":2,"111":2}})
        for path in range(2):
            for basis,bits in [("z","000"),("xy","001"),("yx","100")]:
                manifest["circuits"].append({"kind":"evolution","trajectory":path,"basis":basis,"batch":0,
                    "shots":1,"pub_index":len(observations)})
                observations.append({"outcome_counts":{bits:1}})
        manifest["batches"].append({"start":0,"stop":len(observations),"status":"collected","job_id":"test","observations":observations})
        measured,conditional,_,_=analyze.reconstruct(manifest,source)
        np.testing.assert_allclose(measured["excitation_flows"],.5)
        self.assertIsNone(conditional["excitation_flows"])
        self.assertIsNone(conditional["populations"])

    def test_population_only_cannot_invent_currents_and_incomplete_jobs_fail(self):
        folder,manifest,source=experiment.read_saved(self.template/"manifest.json")
        manifest=copy.deepcopy(manifest)
        manifest["configuration"]["measurement_bases"]=["z"]
        rows=[r for r in manifest["circuits"] if r["basis"]=="z"]
        for i,row in enumerate(rows): row["pub_index"]=i
        manifest["circuits"]=rows
        observations=[{"outcome_counts":{"0"*10:r["shots"]}} for r in rows]
        manifest["batches"]=[{"start":0,"stop":len(rows),"status":"collected","job_id":"test","observations":observations}]
        measured,_,_,_=analyze.reconstruct(manifest,source)
        self.assertEqual(set(measured),{"populations"})
        manifest["batches"][0]["observations"].pop()
        with self.assertRaisesRegex(ValueError,"incomplete"):
            analyze.reconstruct(manifest,source)


if __name__=="__main__":
    unittest.main()
