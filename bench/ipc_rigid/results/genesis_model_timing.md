| scene | frames (equal Newton) | Genesis median ms | CGQ median ms | ratio | equal-Newton median ms G / C | ratio | ms per Newton G / C | ratio | Newton G / C | PCG G / C |
|---|---|---|---|---|---|---|---|---|---|---|
| case 2 box_on_cloth (tol 1e-6) | 149 (142) | 69.97 | 20.73 | 3.37 | 69.92 / 20.59 | 3.40 | 12.56 / 4.06 | 3.10 | 925 / 916 | 10869 / 8306 |
| case 3 Franka press (tol 5e-2) | 98 (15) | 128.28 | 39.17 | 3.27 | 41.90 / 37.24 | 1.13 | 13.55 / 9.50 | 1.43 | 1079 / 536 | 23791 / 10909 |

Rigid preconditioners differ: CGQ's articulated factorization of the forest (a 6x6 block per body, edge pivots) against Genesis's island Cholesky of its rigid block (with the proxy stiffness); the PCG counts are beside the times.
