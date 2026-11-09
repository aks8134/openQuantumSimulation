"""One bounded trajectory/control group, shared by preparation and benchmarking."""
from contextlib import nullcontext

import numpy as np


def compiler_methods(backend):
    """Preserve backend defaults without sending a service object to workers."""
    methods = {}
    for stage in ("translation", "scheduling"):
        getter = getattr(backend, f"get_{stage}_stage_plugin", None)
        if getter is not None:
            value = getter()
            if value is not None:
                methods[f"{stage}_method"] = value
    return methods


def native_template(source, configuration, row, backend, *, target_only=False):
    from Model2.Experiment8 import run_hardware as run
    path = configuration["physical_path"]
    layout = path[1:-1]+[path[0], path[-1]]
    control = row["kind"] == "control"
    logical = run.raw.build_body(source, 0 if control else row["trajectory"], steps=0 if control else None,
                                 populations=row.get("expected_populations"))
    options = (dict(target=backend.target, num_processes=1, **backend.compiler_methods)
               if target_only else dict(backend=backend))
    native = run.transpile(logical, initial_layout=layout, optimization_level=3, seed_transpiler=11, **options)
    if native.layout.final_index_layout(filter_ancillas=True) != layout:
        raise ValueError("transpiler changed the fixed mapping")
    return run.raw.circuits.append_readout(native, layout[:10], "z")


def compile_group(source, configuration, rows, backend, *, job_index, shard_name,
                  templates=None, verbose=True, target_only=False):
    # Import lazily so the CLI and its patchable compiler entry points remain
    # the same for serial preparation, spawned workers and existing tests.
    from Model2.Experiment8 import run_hardware as run

    c = configuration
    templates = {} if templates is None else templates
    key = run.protocol.template_key(rows[0])
    if any(run.protocol.template_key(row) != key for row in rows):
        raise ValueError("a preparation group must contain exactly one trajectory or control")
    control = rows[0]["kind"] == "control"
    layout = c["physical_path"][1:-1]+[c["physical_path"][0], c["physical_path"][-1]]
    policy = dict(c["dd_policy"], physical_qubits=c["physical_path"][1:-1])
    if key not in templates:
        templates[key] = native_template(source, c, rows[0], backend, target_only=target_only)
    programs, windows = [], []
    vector = np.zeros(backend.num_qubits, dtype=np.int64)
    for row in rows:
        variant = run.protocol.make_variant(templates[key], c, row)
        variant.metadata.update(experiment=run.EXPERIMENT, pub_index=row["pub_index"], batch=job_index,
            kind=row["kind"], dd_condition="on", dd_policy=c["dd_policy"])
        variant.metadata["control" if control else "trajectory"] = row["control"] if control else row["trajectory"]
        stage = run.progress(f"Scheduling DD-on job {job_index+1}, {row['case_id']}") if verbose else nullcontext()
        with stage:
            local, original = run.timing.schedule(variant, backend.target, block_ordering_callable=run.flat_block_order)
            program, events = run.timing.schedule(variant, backend.target, policy=policy,
                                                 block_ordering_callable=run.flat_block_order)
            check = run.timing.verify_pair(local, program, original, events, policy,
                                           alignment_dt=int(backend.target.pulse_alignment or 1))
        row.update(isa_file=shard_name, isa_index=len(programs), metrics=run.raw.circuits.metrics(program),
            final_logical_to_physical=layout, dd_check={k:v for k,v in check.items() if k != "windows"})
        if c["capture_timing"]:
            windows.append(dict(pub_index=row["pub_index"], case_id=row["case_id"], **check))
        run.base.validate_circuits({"configuration":c}, [program], [row], backend,
                                   expected_experiment=run.EXPERIMENT)
        run.validate_isa_circuits([program], backend.target)
        programs.append(program)
        vector += run.instruction_audit.footprint(program)
        del local, original, events, variant
    if not control:
        del templates[key]
    return programs, windows, vector
