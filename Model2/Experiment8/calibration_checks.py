"""Record fixed-path calibration diagnostics without altering a hardware plan."""
from __future__ import annotations

from datetime import datetime, timezone


def settings():
    return dict(avoid_qubits=[], min_t1_seconds=80e-6, min_t2_seconds=60e-6,
                max_cz_error=.005, max_readout_error=.02, max_expansions=2_000_000)


def record(runner, backend, manifest, folder, *, stage):
    c = manifest["configuration"]
    path = list(c["physical_path"] or c["requested_physical_path"])
    limits = settings()
    invalid = [q for q in path if q < 0 or q >= backend.num_qubits]
    connectivity_error = None
    try:
        runner.raw.circuits.chain_layout(backend, 10, "boundary", path)
    except ValueError as error:
        connectivity_error = str(error)
    if invalid:
        values = dict(qubits={}, edges=[], threshold_failures=[dict(qubit=q, quantity="physical_qubit_index",
            value=q, threshold=backend.num_qubits, condition="valid index") for q in invalid])
    else:
        values = runner.base.calibration_observations(backend, path, limits)
    for q in set(path) & set(limits["avoid_qubits"]):
        values["threshold_failures"].append(dict(qubit=q, quantity="avoid_qubits", value=q,
                                                threshold=limits["avoid_qubits"], condition="not listed"))
    stamp = datetime.now(timezone.utc)
    report = dict(artifact_type="model2_dd_path_calibration_check_v1", stage=stage,
        captured_at_utc=stamp.isoformat(), backend=backend.name, plan_sha256=manifest["plan_sha256"],
        physical_path=path, saved_policy=c["calibration_policy"], thresholds=limits,
        connectivity_passed=connectivity_error is None, connectivity_error=connectivity_error,
        calibration_screen_passed=not values["threshold_failures"] and connectivity_error is None,
        **values, scope="Current calibration observations only. No simulation, compilation, controls or QPU jobs. "
                        "The existing manifest, mapping, policy, thresholds and job IDs are unchanged.")
    report["sha256"] = runner.input_data.digest(report)
    destination = folder/"calibration_checks"/f"calibration_{stamp.strftime('%Y%m%dT%H%M%S_%fZ')}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    runner.atomic_json(destination, report)
    print(f"Calibration report: {destination}", flush=True)
    print(f"Fixed path: {path}; saved policy: {c['calibration_policy']}; "
          f"current screen {'PASSED' if report['calibration_screen_passed'] else 'FAILED'}", flush=True)
    for failure in report["threshold_failures"]:
        label = f"qubit {failure['qubit']}" if "qubit" in failure else f"CZ {failure['qubits']}"
        quantity, value, threshold = failure["quantity"], failure["value"], failure["threshold"]
        if quantity in ("t1_seconds", "t2_seconds"):
            shown = "unavailable" if value is None else f"{value*1e6:.3f} us"
            bound = f"{threshold*1e6:g} us"
        elif quantity in ("readout_error", "cz_error"):
            shown = "unavailable" if value is None else f"{100*value:.6f}%"
            bound = f"{100*threshold:g}%"
        else:
            shown, bound = str(value), str(threshold)
        print(f"  FAILED {label}: {quantity} = {shown}; {failure['condition']} {bound}", flush=True)
    if connectivity_error:
        print(f"  FAILED connectivity: {connectivity_error}", flush=True)
    return destination, report
