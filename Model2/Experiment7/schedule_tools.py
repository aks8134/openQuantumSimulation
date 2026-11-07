"""Compiler timing inspection and conservative, system-only XX insertion.

The visible schedules come exclusively from IBM result metadata. Local dynamic
scheduling is used to construct and check circuits, never as a substitute plot.
"""
from __future__ import annotations

from collections import Counter
import csv
import io
from pathlib import Path
import re

import numpy as np
from qiskit.circuit.library import XGate
from qiskit.circuit import Delay
from qiskit.transpiler import PassManager
from qiskit_ibm_runtime.transpiler.passes.scheduling import (
    ALAPScheduleAnalysis, PadDelay, PadDynamicalDecoupling,
)
from qiskit_ibm_runtime.transpiler.passes.scheduling.utils import block_order_op_nodes

DD_LABEL = "model2_experiment7_dd"


class SystemDD(PadDynamicalDecoupling):
    """Also leave post-readout waits alone; there is no remaining state to protect.

    This small adapter uses Runtime's padding hook; covered by compatibility
    tests because the experimental dynamic scheduler can change between releases.
    """
    def __init__(self, *args, final_bits=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.final_bits = set(final_bits)

    def _pre_runhook(self, dag):
        super()._pre_runhook(dag)
        # Runtime 0.49 and 0.51 cache Target instruction durations (seconds)
        # directly while the padding algorithm operates in dt. Recompute the
        # cache in dt from the unchanged real target; do not invent durations or
        # mutate backend calibration. This also works if the SDK fixes its cache.
        for qubit, sequences in self._dd_sequence_lengths.items():
            physical = dag.qubits.index(qubit)
            for i, _ in enumerate(sequences):
                values = [int(self._target.seconds_to_dt(self._target[g.name][(physical,)].duration))
                          for g in self._dd_sequences[i]]
                if any(v <= 0 for v in values):
                    raise ValueError("DD pulse duration must be positive in dt")
                self._dd_sequence_lengths[qubit][i] = values

    def _pad(self, block_idx, qubit, t_start, t_end, next_node, prev_node, enable_dd=False):
        if (getattr(prev_node, "name", None) == "measure"
                and set(prev_node.cargs) & self.final_bits):
            self._apply_scheduled_op(block_idx, t_start, Delay(t_end-t_start, "dt"), qubit)
            return
        super()._pad(block_idx, qubit, t_start, t_end, next_node, prev_node, enable_dd)


def timing_model(backend, path):
    """Fingerprint the durations/constraints needed to reproduce the scout."""
    target = backend.target
    dt = target.dt
    if dt is None or not np.isfinite(dt) or dt <= 0:
        raise ValueError("backend must supply a positive dt")
    instructions = []
    for name in ("rz", "sx", "x", "cz", "reset", "measure"):
        if name not in target.operation_names:
            raise ValueError(f"backend must support {name}")
        for qargs, properties in target[name].items():
            if qargs is not None and set(qargs) <= set(path):
                duration = getattr(properties, "duration", None)
                if duration is None or not np.isfinite(duration) or duration < 0:
                    raise ValueError(f"missing/invalid duration: {name}{qargs}")
                instructions.append(dict(name=name, qubits=list(qargs), duration_seconds=float(duration)))
    return dict(dt_seconds=float(dt),
        constraints={name: int(getattr(target, name, 1) or 1) for name in
                     ("pulse_alignment", "acquire_alignment", "granularity", "min_length")},
        instructions=sorted(instructions, key=lambda row: (row["name"], row["qubits"])))


def _duration(node, circuit, target):
    if node.name == "barrier":
        return 0
    if node.name == "delay":
        if node.op.unit != "dt":
            raise ValueError("scheduled delays must use dt")
        return int(node.op.duration)
    qargs = tuple(circuit.find_bit(q).index for q in node.qargs)
    return int(target.seconds_to_dt(target[node.name][qargs].duration))


def schedule(circuit, target, *, policy=None):
    """Same ALAP policy for both arms; no optimization after DD insertion."""
    # Both passes visit the identical input DAG. IBM's ordering walks many
    # descendants in deep reset circuits; retain that exact order for the second
    # visit rather than recomputing it. Cache lifetime is one schedule call.
    orders = {}
    def ordering(dag):
        key = id(dag)
        if key not in orders:
            orders[key] = (dag, list(block_order_op_nodes(dag)))
        return iter(orders[key][1])
    if policy is None:
        padder = PadDelay(target=target, schedule_idle_qubits=False, block_ordering_callable=ordering)
    else:
        padder = SystemDD(target=target,
            dd_sequences=[XGate(label=DD_LABEL), XGate(label=DD_LABEL)],
            qubits=policy["physical_qubits"], spacings=[[.25, .5, .25]],
            sequence_min_length_ratios=[policy["min_idle_ratio"]],
            pulse_alignment=int(target.pulse_alignment or 1),
            skip_reset_qubits=True, insert_multiple_cycles=False, schedule_idle_qubits=False,
            block_ordering_callable=ordering,
            final_bits=next((list(r) for r in circuit.cregs if r.name == "outcome"), []))
    manager = PassManager([ALAPScheduleAnalysis(target=target, block_ordering_callable=ordering), padder])
    output = manager.run(circuit)
    events = []
    for node, (block, start) in manager.property_set["node_start_time"].items():
        events.append(dict(name=node.name, qubits=[circuit.find_bit(q).index for q in node.qargs],
            clbits=[circuit.find_bit(c).index for c in node.cargs],
            params=[float(x) for x in node.op.params], block=int(block), start_dt=int(start),
            duration_dt=_duration(node, circuit, target), dd=node.op.label == DD_LABEL))
    return output, events


def verify_pair(baseline, decoupled, baseline_events, dd_events, policy, *, alignment_dt=1):
    """Check every original operation's local block/time, plus each XX window.

    This is a construction check. Actual controller timing is checked separately
    using the two server-returned schedules, including reset/feedback latency.
    """
    def anchors(events):
        return Counter((e["name"], tuple(e["qubits"]), tuple(e["clbits"]), tuple(e["params"]),
                        e["block"], e["start_dt"], e["duration_dt"])
                       for e in events if not e["dd"] and e["name"] != "delay")
    if anchors(baseline_events) != anchors(dd_events):
        raise ValueError("DD moved or changed an original scheduled operation")
    if baseline.num_clbits != decoupled.num_clbits:
        raise ValueError("DD changed the classical registers")
    windows, pulses = [], [e for e in dd_events if e["dd"]]
    used = set()
    for gap in (e for e in baseline_events if e["name"] == "delay"):
        qubit, start, stop = gap["qubits"][0], gap["start_dt"], gap["start_dt"]+gap["duration_dt"]
        indices = [i for i, e in enumerate(pulses) if e["qubits"] == [qubit]
                   and e["block"] == gap["block"] and start <= e["start_dt"] < stop]
        if not indices:
            continue
        seq = sorted((pulses[i] for i in indices), key=lambda e: e["start_dt"])
        if (qubit not in policy["physical_qubits"] or len(seq) != 2
                or any(e["name"] != "x" or e["duration_dt"] <= 0
                       or e["start_dt"] % alignment_dt or e["start_dt"]+e["duration_dt"] > stop for e in seq)
                or seq[0]["start_dt"]+seq[0]["duration_dt"] > seq[1]["start_dt"]
                or gap["duration_dt"] <= policy["min_idle_ratio"]*sum(e["duration_dt"] for e in seq)):
            raise ValueError("DD did not form one fitting XX identity inside an eligible original gap")
        used.update(indices)
        windows.append(dict(physical_qubit=qubit, block=gap["block"], start_dt=start,
            duration_dt=gap["duration_dt"], pulse_start_dt=[e["start_dt"] for e in seq]))
    if len(used) != len(pulses):
        raise ValueError("DD pulse outside an original scheduled idle window")
    return dict(added_x_pulses=len(pulses), protected_windows=len(windows),
        added_x_by_qubit={str(q): sum(e["qubits"] == [q] for e in pulses)
                          for q in sorted({e["qubits"][0] for e in pulses})},
        protected_idle_dt=sum(w["duration_dt"] for w in windows), windows=windows,
        original_operations_and_local_times_preserved=True,
        scope="Local block check only; not a measured or compiler-returned wall-clock check.")


def timing_text(value):
    # Documentation uses both a newline-separated string and a list of strings.
    if isinstance(value, list) and all(isinstance(line, str) for line in value):
        value = "\n".join(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("missing IBM compilation.scheduler_timing.timing; no local timing fallback")
    return value


def parse_timing(value):
    rows = []
    for line in csv.reader(io.StringIO(timing_text(value))):
        if not line or not any(x.strip() for x in line):
            continue
        if len(line) != 6:
            raise ValueError(f"unrecognized IBM timing row with {len(line)} columns")
        branch, instruction, channel, start, duration, pulse = [x.strip() for x in line]
        start, duration = int(start), int(duration)
        if min(start, duration) < 0:
            raise ValueError("negative compiler timing")
        rows.append(dict(branch=branch, instruction=instruction, channel=channel,
                         start_dt=start, duration_dt=duration, pulse=pulse))
    if not rows:
        raise ValueError("empty compiler timing")
    return rows


def gap_table(rows, path, dt_seconds):
    """Unoccupied drive-channel intervals, NOT automatically legal DD windows.

    Controller/readout occupancy is retained as context. No leading/trailing
    interval is called idle: state preparation/FINI may be implicit. Branches
    are kept separate; alternatives are never merged into a fictitious clock.
    """
    output = []
    control = [r for r in rows if not re.fullmatch(r"Qubit \d+", r["channel"])]
    for branch in sorted({r["branch"] for r in rows}):
        for qubit in path:
            active = sorted((r for r in rows if r["branch"] == branch
                and r["channel"] == f"Qubit {qubit}" and r["duration_dt"] > 0
                and r["pulse"] != "delay" and not r["instruction"].lower().startswith("delay")),
                key=lambda r: (r["start_dt"], r["duration_dt"]))
            end, previous = None, None
            for row in active:
                if end is not None and row["start_dt"] > end:
                    stop = row["start_dt"]
                    overlap = [r for r in control if r["branch"] in (branch, "main")
                        and r["start_dt"] < stop and r["start_dt"]+r["duration_dt"] > end]
                    output.append(dict(branch=branch, physical_qubit=qubit,
                        role="system" if qubit in path[1:-1] else "bath", start_dt=end, stop_dt=stop,
                        duration_dt=stop-end, duration_us=(stop-end)*dt_seconds*1e6,
                        previous_instruction=previous, next_instruction=row["instruction"],
                        controller_or_readout_overlap=";".join(sorted({r["channel"] for r in overlap})),
                        dd_eligibility="requires local block, alignment and state-lifetime check"))
                if end is None or row["start_dt"]+row["duration_dt"] >= end:
                    end, previous = row["start_dt"]+row["duration_dt"], row["instruction"]
    return output


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def draw(value, destination, *, channels=None):
    try:
        from qiskit_ibm_runtime.visualization import draw_circuit_schedule_timing
    except (ImportError, ModuleNotFoundError) as error:
        raise ValueError("timing HTML needs plotly: install Experiment7/requirements.txt; raw timing is retained") from error
    figure = draw_circuit_schedule_timing(circuit_schedule=timing_text(value),
        included_channels=channels, filter_readout_channels=False, filter_barriers=False, width=1500)
    figure.write_html(str(destination), include_plotlyjs=True, full_html=True)


def server_comparison(baseline_rows, dd_rows, *, expected_added_x_by_qubit=None):
    """Compare protected non-X operation/controller timestamps without guessing
    which compiler X pulses implement reset, original gates, or inserted DD.
    """
    def anchors(rows):
        return Counter((r["branch"], r["instruction"], r["channel"], r["pulse"],
                        r["start_dt"], r["duration_dt"]) for r in rows
                       if not re.match(r"^(x|delay)(_|$)", r["instruction"].lower())
                       and r["instruction"].lower() != "barrier")
    left, right = anchors(baseline_rows), anchors(dd_rows)
    span = lambda rows: max(r["start_dt"]+r["duration_dt"] for r in rows)
    def x_counts(rows):
        counts = Counter()
        for row in rows:
            channel = re.fullmatch(r"Qubit (\d+)", row["channel"])
            if channel and re.match(r"^x(_|$)", row["instruction"].lower()) and row["pulse"] == "play":
                counts[channel.group(1)] += 1
        return counts
    base_x, dd_x = x_counts(baseline_rows), x_counts(dd_rows)
    delta_x = {q: dd_x[q]-base_x[q] for q in sorted(base_x.keys() | dd_x.keys()) if dd_x[q] != base_x[q]}
    result = dict(non_x_anchor_timing_equal=left == right, baseline_end_dt=span(baseline_rows),
        dd_end_dt=span(dd_rows), removed_or_moved_anchors=sum((left-right).values()),
        added_or_moved_anchors=sum((right-left).values()),
        observed_added_x_by_qubit=delta_x,
        scope="Conservative service comparison including controller/readout and FINI; differences require inspection. X identity/placement is checked locally, not inferred from pulse names.")
    if expected_added_x_by_qubit is not None:
        result.update(expected_added_x_by_qubit=expected_added_x_by_qubit,
                      added_x_counts_match=delta_x == expected_added_x_by_qubit)
    return result
