"""Bell-pair experiment defined entirely by an upstream consumer."""

from IBMRuntime import (
    Aer,
    CompilerConfig,
    Err,
    Ok,
    Sample,
    counts_dict,
    cx,
    empty,
    execution_plan,
    h,
    measure,
    pipe,
    run_sync,
)


def bell_circuit():
    return pipe(
        empty(2, 2, name="bell_pair"),
        h(0),
        cx(0, 1),
        measure(0, 0),
        measure(1, 1),
    )


def bell_plan(
    target=Aer(),
    *,
    shots: int = 1024,
    seed_transpiler: int | None = 0,
    seed_simulator: int | None = 0,
):
    return execution_plan(
        circuit=bell_circuit(),
        target=target,
        compiler=CompilerConfig(seed_transpiler=seed_transpiler),
        workload=Sample(shots=shots, seed_simulator=seed_simulator),
    )


def main() -> int:
    match run_sync(bell_plan()):
        case Ok(sample):
            print(counts_dict(sample))
            return 0
        case Err(error):
            print(error)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
