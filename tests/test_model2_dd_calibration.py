"""Fixed-path diagnostics and preparation screening without IBM credentials."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from qiskit.circuit import Measure, Parameter, Reset
from qiskit.circuit.library import CZGate, RZGate, SXGate, XGate
from qiskit.transpiler import InstructionProperties, QubitProperties, Target

from Model2.Experiment8 import run_hardware as run


def backend(failing=True):
    properties = [QubitProperties(t1=200e-6, t2=150e-6) for _ in range(12)]
    if failing:
        properties[3] = QubitProperties(t1=50e-6, t2=40e-6)
    target = Target(num_qubits=12, dt=4e-9, qubit_properties=properties,
                    pulse_alignment=1, acquire_alignment=1, granularity=1, min_length=2)
    for gate, seconds, error in ((RZGate(Parameter("theta")), 0., 0.), (SXGate(), 40e-9, .001),
                                  (XGate(), 40e-9, .001), (Reset(), 400e-9, None),
                                  (Measure(), 1.94e-6, .005)):
        values = {(q,):InstructionProperties(duration=seconds,
                  error=.03 if failing and gate.name == "measure" and q == 3 else error) for q in range(12)}
        target.add_instruction(gate, values)
    edges = [(q,q+1) for q in range(11)]+[(q+1,q) for q in range(11)]
    target.add_instruction(CZGate(), {pair:InstructionProperties(duration=68e-9,
        error=.006 if failing and set(pair) == {7,8} else .001) for pair in edges})
    return SimpleNamespace(name="calibration_test", num_qubits=12, target=target,
                           coupling_map=target.build_coupling_map())


def plan(root, policy="screen"):
    reference = dict(configuration=dict(backend="calibration_test", physical_path=list(range(12))))
    opts = run.parse_arguments(["plan", "--backend", "calibration_test", "--trajectory-count", "2",
        "--shots", "8", "--control-shots", "128", "--batch-size", "41", "--bootstrap", "100",
        "--calibration-policy", policy, "--output-directory", str(root)])
    with patch.object(run, "read_reference", return_value=reference):
        folder, manifest, _ = run.new_plan(opts)
    return folder, manifest


def test_inspection_reports_exact_values_and_preserves_plan(tmp_path, capsys):
    folder, m = plan(tmp_path)
    before = (folder/"manifest.json").read_bytes()
    opts = run.parse_arguments(["inspect-calibration", "--manifest", str(folder/"manifest.json")])
    with patch.object(run.raw.runtime, "make_service", side_effect=AssertionError("no IBM account")), \
         patch.object(run, "transpile", side_effect=AssertionError("inspection cannot compile")):
        report = run.inspect_calibration(opts, backend=backend())
    assert not report["calibration_screen_passed"] and report["connectivity_passed"]
    failures = {(f.get("qubit"),f["quantity"]):f for f in report["threshold_failures"]}
    assert failures[(3,"t1_seconds")]["value"] == 50e-6
    assert failures[(3,"t2_seconds")]["threshold"] == 60e-6
    assert failures[(3,"readout_error")]["value"] == .03
    assert any(f["quantity"] == "cz_error" for f in report["threshold_failures"])
    assert (folder/"manifest.json").read_bytes() == before
    assert not list(folder.glob("*.qpy"))
    assert len(list((folder/"calibration_checks").glob("*.json"))) == 1
    text = capsys.readouterr().out
    assert "50.000 us" in text and "3.000000%" in text


def test_failed_prepare_saves_diagnostics_before_any_compilation(tmp_path):
    folder, _ = plan(tmp_path)
    before = (folder/"manifest.json").read_bytes()
    opts = run.parse_arguments(["prepare", "--resume", str(folder/"manifest.json")])
    with patch.object(run.raw.runtime, "make_service", side_effect=AssertionError("no account")), \
         patch.object(run, "transpile", side_effect=AssertionError("failed quality screen must stop before compilation")):
        with pytest.raises(ValueError, match="current calibration screen"):
            run.prepare(opts, backend=backend())
    assert before == (folder/"manifest.json").read_bytes()
    assert not list(folder.glob("*.qpy"))
    saved = json.loads(next((folder/"calibration_checks").glob("*.json")).read_text())
    assert saved["saved_policy"] == "screen" and not saved["calibration_screen_passed"]


@pytest.mark.parametrize("policy,failing", [("screen",False), ("record-only",True)])
def test_valid_screen_and_explicit_record_only_reach_compilation(tmp_path, policy, failing):
    folder, _ = plan(tmp_path, policy)
    opts = run.parse_arguments(["prepare", "--resume", str(folder/"manifest.json")])
    with patch.object(run.raw.runtime, "make_service", side_effect=AssertionError("no account")), \
         patch.object(run, "transpile", side_effect=RuntimeError("compilation reached")):
        with pytest.raises(RuntimeError, match="compilation reached"):
            run.prepare(opts, backend=backend(failing))
    saved = json.loads(next((folder/"calibration_checks").glob("*.json")).read_text())
    assert saved["saved_policy"] == policy
    assert saved["calibration_screen_passed"] == (not failing)
