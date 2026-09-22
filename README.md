# Open Quantum Systems Simulation

## Functional IBM Runtime wrapper

`IBMRuntime` is an OCaml-inspired Python API. Circuits, targets, execution
plans, errors, and results are immutable values. Qiskit objects exist only in
the interpreter boundary.

Run the upstream Bell-pair example on Aer:

```bash
.venv/bin/python -m examples.bell_pair
```

Or use the public API:

```python
from IBMRuntime import Err, Ok, counts_dict, run_sync
from examples.bell_pair import bell_plan

match run_sync(bell_plan(shots=1024)):
    case Ok(sample):
        print(counts_dict(sample))
    case Err(error):
        print(error)
```

The same immutable plan has an asynchronous interpreter:

```python
from IBMRuntime import run_async
from examples.bell_pair import bell_plan

result = await run_async(bell_plan(shots=1024))
```

Estimator workloads use an immutable Pauli-observable algebra. The Qiskit
interpreter handles observable lowering, transpiler-layout application, and
the provider-specific Estimator primitive:

```python
from IBMRuntime import (
    Aer,
    Estimate,
    Ok,
    empty,
    execution_plan,
    h,
    observable,
    pauli,
    pauli_term,
    pipe,
    run_sync,
)

x_observable = observable("x", pauli_term(1.0, pauli("X", 0)))
plan = execution_plan(
    pipe(empty(1, 0), h(0)),
    Aer(),
    workload=Estimate((x_observable,), precision=0.02),
)

match run_sync(plan):
    case Ok(result):
        print(result.values)
```

For IBM hardware, load the account explicitly and supply an immutable runtime
environment:

```python
from IBMRuntime import (
    Err,
    IBMHardware,
    Ok,
    RuntimeEnvironment,
    load_ibm_account,
    run_sync,
)
from examples.bell_pair import bell_plan

match load_ibm_account("IBMRuntime/apikey.json"):
    case Err(error):
        result = Err(error)
    case Ok(account):
        result = run_sync(
            bell_plan(IBMHardware("YOUR_BACKEND_NAME"), shots=1024),
            RuntimeEnvironment(ibm_account=account),
        )
```

Run the test suite with:

```bash
.venv/bin/python -m unittest discover -s tests -v
```
