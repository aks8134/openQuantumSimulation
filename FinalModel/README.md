# FinalModel

Independent studies combining Model1, Model2 and Model3. The dynamics run
locally. Phoenix resource compilation reads backend metadata once, then
works from a frozen local target. No hardware jobs are submitted.
Existing experiments and their saved results are not modified.

Start with [Experiment1: scaling to 100 system qubits](Experiment1/Experiment1.md).
The [results](Experiment1/Results.md) cover the full offline study with
J=gamma=1 and H=(J/2) sum(XX+YY) in all three models.

[Phoenix resource compilation](Experiment1/PhoenixResources.md) recompiles
the existing schedules against a captured real target without QPU jobs.
