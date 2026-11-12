# IPC formula parity: et-fix

Per case and input set: rows, then per backend the failing rows, the largest real error over the scale of its output (max |g - c| / s), and the rows that passed by the order clause and by the referee (compare.py).

| case | function | set | rows | cuda_fast-math-false_et-fix failures | cuda_fast-math-false_et-fix max err / s | cuda_fast-math-false_et-fix order rows | cuda_fast-math-false_et-fix referee rows | cpu_fast-math-false_et-fix failures | cpu_fast-math-false_et-fix max err / s | cpu_fast-math-false_et-fix order rows | cpu_fast-math-false_et-fix referee rows |
|---|---|---|---|---|---|---|---|---|---|---|---|
| intersect | edge-triangle intersection | degenerate | 800 | 0 | - | 0 | 0 | 0 | - | 0 | 0 |
| intersect | edge-triangle intersection | random | 4000 | 0 | - | 0 | 0 | 0 | - | 0 | 0 |
| intersect | edge-triangle intersection | regression | 8 | 0 | - | 0 | 0 | 0 | - | 0 | 0 |

Totals: cuda_fast-math-false_et-fix: 0 failing rows of 4808, cpu_fast-math-false_et-fix: 0 failing rows of 4808
