# Resources scaling results

See Experiment1.md for definitions, assumptions and commands.

| model | N | family | status | path | topology | logical_exchanges | native_depth | two_qubit_depth | native_cx | T_estimate |
|---|---|---|---|---|---|---|---|---|---|---|
| model1 | 2 | constant | compiled | 0 | all_to_all | 62 | 460 | 124 | 124 | 7068 |
| model1 | 2 | constant | compiled | 0 | line | 62 | 460 | 124 | 124 | 7068 |
| model1 | 2 | warped_cosine | compiled | 0 | all_to_all | 62 | 460 | 124 | 124 | 7068 |
| model1 | 2 | warped_cosine | compiled | 0 | line | 62 | 460 | 124 | 124 | 7068 |
| model1 | 2 | fixed | no_validated_schedule | — | — | — | — | — | — | — |
| model2 | 2 | constant | compiled | 0 | all_to_all | 12 | 95 | 24 | 24 | 1128 |
| model2 | 2 | constant | compiled | 0 | line | 12 | 95 | 24 | 24 | 1128 |
| model2 | 2 | warped_cosine | compiled | 0 | all_to_all | 12 | 95 | 24 | 24 | 1128 |
| model2 | 2 | warped_cosine | compiled | 0 | line | 12 | 95 | 24 | 24 | 1128 |
| model2 | 2 | fixed | compiled | 0 | all_to_all | 58 | 438 | 116 | 116 | 6264 |
| model2 | 2 | fixed | compiled | 0 | line | 58 | 438 | 116 | 116 | 6264 |
| model3 | 2 | constant | compiled | 0 | all_to_all | 156 | 1149 | 312 | 312 | 19032 |
| model3 | 2 | constant | compiled | 0 | line | 156 | 1149 | 312 | 312 | 19032 |
| model3 | 2 | warped_cosine | compiled | 0 | all_to_all | 50 | 372 | 100 | 100 | 5600 |
| model3 | 2 | warped_cosine | compiled | 0 | line | 50 | 372 | 100 | 100 | 5600 |
| model3 | 2 | fixed | compiled | 0 | all_to_all | 156 | 1149 | 312 | 312 | 19032 |
| model3 | 2 | fixed | compiled | 0 | line | 156 | 1149 | 312 | 312 | 19032 |
