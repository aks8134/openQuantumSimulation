"""Native qem transforms for Experiment 11, without global experiment hooks."""
from dataclasses import asdict

from qiskit.circuit import Gate
from qiskit.transpiler import InstructionDurations

import qem
from .analysis import qem_value

VIRTUAL_NAMES = ("rz", "p", "u1", "phase")


def _inverse_factory(target):
    def inverse(gate):
        if gate.name == "sx" and target is not None and "sxdg" not in target.operation_names:
            from qiskit.circuit.library import SXGate
            return (SXGate(), SXGate(), SXGate())
        return (gate.inverse(),)
    return inverse


def _readout_model(circuit, config):
    if config["no_readout_mitigation"]:
        return None
    mapping = {}
    for item in circuit.data:
        if item.operation.name == "measure":
            physical = circuit.find_bit(item.qubits[0]).index
            classical = circuit.find_bit(item.clbits[0]).index
            if physical in mapping or classical in mapping.values():
                raise ValueError("terminal measurements must be unique")
            mapping[physical] = classical
    calibration = config["readout_calibration"]
    if calibration is None:
        if config["backend"] != "aer":
            raise ValueError("noisy fake/hardware readout mitigation needs explicit calibration")
        return qem.ReadoutModel(tuple(qem.ReadoutBlock((bit,), ((1.0, 0.0), (0.0, 1.0)))
                                     for bit in sorted(mapping.values())),
                                source="identity: noiseless Aer, not measured calibration")
    blocks = []
    for block in calibration["blocks"]:
        physicals = tuple(block["physical_qubits"])
        if not set(physicals).intersection(mapping):
            continue
        if not all(q in mapping for q in physicals):
            raise ValueError("correlated calibration block is only partially measured")
        blocks.append(qem.ReadoutBlock(tuple(mapping[q] for q in physicals),
                                      tuple(tuple(row) for row in block["assignment"])))
    model = qem.ReadoutModel(tuple(blocks), source=calibration["source"])
    if set(model.bits) != set(mapping.values()):
        raise ValueError("calibration does not cover all actually measured physical qubits")
    return model


def transform(native, target, config, *, base_index, scale_index, repetition):
    scale = config["zne_scale_factors"][scale_index]
    serial = ((base_index * len(config["zne_scale_factors"]) + scale_index)
              * config["fold_repetitions"] + repetition)
    seed = config["seed_folding"] + serial
    base = qem.circuit_metrics(native)
    eligible = sum(isinstance(item.operation, Gate) and item.operation.name not in VIRTUAL_NAMES
                   for item in native.data)
    if scale == 1:
        folded = native
        folds = 0
    else:
        batch = qem_value(qem.fold(native, qem.FoldPlan(scale, seed=seed,
                              excluded_names=VIRTUAL_NAMES), target=target,
                              inverse_factory=_inverse_factory(target)))
        folded = batch.circuit
        folds = sum(batch.selection.repetitions)
    folded_metrics = qem.circuit_metrics(folded)
    inserted = folded_metrics.nonvirtual_gates - base.nonvirtual_gates
    if config["no_pauli_twirling"]:
        twirled, twirling = folded, None
    else:
        batch = qem_value(qem.twirl(folded, qem.TwirlingPlan(
            seed=config["seed_twirling"] + serial,
            gate_names=("cx", "cz", "rxx", "ryy", "rzz")), target=target))
        twirled = batch.circuit
        twirling = {"seed": batch.seed, "twirled_gate_count": len(batch.choices),
                    "untwirled_two_qubit_gates": batch.untwirled_two_qubit_gates}
    twirled_metrics = qem.circuit_metrics(twirled)
    dd = None
    if config["no_dynamical_decoupling"]:
        submitted = twirled
    else:
        active = tuple(sorted({twirled.find_bit(bit).index for item in twirled.data
                               if item.operation.name not in ("barrier", "delay") for bit in item.qubits}))
        data = config["dd_durations"]
        durations = None if data is None else InstructionDurations(data["instructions"], dt=data["dt"])
        batch = qem_value(qem.apply_dd(twirled, qem.DDPlan(sequence=config["dd_sequence"],
            spacing=None if config["dd_spacing_fractions"] is None else tuple(config["dd_spacing_fractions"]),
            qubits=active), target=target if durations is None else None, durations=durations))
        submitted = batch.circuit
        dd = {"sequence": batch.sequence, "inserted_gate_count": batch.inserted_gate_count,
              "pulse_alignment": batch.pulse_alignment, "acquire_alignment": batch.acquire_alignment}
        for item in submitted.data:
            if item.operation.name != "barrier" and not target.instruction_supported(
                operation_name=item.operation.name,
                qargs=tuple(submitted.find_bit(bit).index for bit in item.qubits),
                parameters=item.operation.params):
                raise ValueError(f"DD emitted unsupported instruction {item.operation.name}")
    readout = _readout_model(submitted, config)
    layout = native.layout
    mapping = (list(range(native.num_qubits)) if layout is None
               else list(map(int, layout.initial_index_layout(filter_ancillas=True))))
    diagnostic = {"base": asdict(base), "post_folding": asdict(folded_metrics),
                  "post_twirling": asdict(twirled_metrics), "submitted": asdict(qem.circuit_metrics(submitted)),
                  "requested_scale_factor": scale,
                  "actual_physical_scale_factor": 1.0 + inserted / eligible if eligible else 1.0,
                  "folded_gate_occurrences": folds, "inserted_physical_gate_count": inserted,
                  "eligible_physical_gate_count": eligible, "fold_seed": seed,
                  "twirling": twirling, "dd": dd, "logical_to_physical": mapping,
                  "measurement_mapping": [(submitted.find_bit(item.qubits[0]).index,
                                           submitted.find_bit(item.clbits[0]).index)
                                          for item in submitted.data if item.operation.name == "measure"],
                  "readout_model": None if readout is None else {
                      "source": readout.source, "blocks": [asdict(block) for block in readout.blocks]}}
    return submitted, diagnostic
