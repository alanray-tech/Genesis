## Scenes

| scene | verts | surf verts | edges | faces | bodies | frames | CGQ candidates / frame | active / frame |
|---|---|---|---|---|---|---|---|---|
| two_bunnies | 3738 | 2156 | 6456 | 4304 | 2 | 30 | 102 | 49 |
| cloth_stack | 8940 | 8940 | 26106 | 17204 | 21 | 30 | 12062 | 7901 |
| wrecking_ball | 12511 | 10438 | 27954 | 18636 | 575 | 30 | 16934 | 16735 |
| kimono | 26086 | 26085 | 76433 | 50380 | 20 | 30 | 24655 | 7195 |
| cloth_stack_x3 | 77540 | 77540 | 230706 | 153204 | 21 | 30 | 598001 | 584279 |

## Correctness

| scene | active sets identical (port) | identical (CGQ kernels on bvh.py candidates) | CGQ candidates all found | candidates only in bvh.py (all frames) |
|---|---|---|---|---|
| two_bunnies | True | True | True | 0 |
| cloth_stack | True | True | True | 1 |
| wrecking_ball | True | True | True | 23 |
| kimono | True | True | True | 14769 |
| cloth_stack_x3 | True | True | True | 5 |

## Timing, median over frames of the per-frame median (p90), ms, bvh.py deferred

| scene | stage | CGQ | bvh.py | bvh.py / CGQ |
|---|---|---|---|---|
| two_bunnies | build | 0.193 (0.194) | 0.513 (0.585) | 2.7x |
| two_bunnies | PT + EE query | 0.121 (0.122) | 0.342 (0.383) | 2.8x |
| two_bunnies |   PT query | | 0.157 (0.171) | |
| two_bunnies |   EE query | | 0.188 (0.209) | |
| two_bunnies | PH query | 0.000 (0.000) | 0.002 (0.003) |  |
| two_bunnies | narrow | 0.034 (0.051) | 0.063 (0.093) | 1.9x |
| two_bunnies | count readback | | 0.205 (0.253) | |
| two_bunnies | total | 0.362 (0.381) | 1.012 (1.151) | 2.8x |
| cloth_stack | build | 0.223 (0.225) | 0.957 (1.954) | 4.3x |
| cloth_stack | PT + EE query | 0.559 (0.563) | 1.419 (1.521) | 2.5x |
| cloth_stack |   PT query | | 0.376 (0.421) | |
| cloth_stack |   EE query | | 1.045 (1.103) | |
| cloth_stack | PH query | 0.000 (0.000) | 0.003 (0.007) |  |
| cloth_stack | narrow | 0.050 (0.051) | 0.078 (0.172) | 1.6x |
| cloth_stack | count readback | | 0.208 (0.417) | |
| cloth_stack | total | 0.835 (0.842) | 2.552 (3.444) | 3.1x |
| wrecking_ball | build | 0.220 (0.223) | 1.014 (2.045) | 4.6x |
| wrecking_ball | PT + EE query | 0.210 (0.211) | 0.468 (0.545) | 2.2x |
| wrecking_ball |   PT query | | 0.188 (0.227) | |
| wrecking_ball |   EE query | | 0.287 (0.318) | |
| wrecking_ball | PH query | 0.007 (0.007) | 0.056 (0.136) | 7.8x |
| wrecking_ball | narrow | 0.049 (0.068) | 0.074 (0.167) | 1.5x |
| wrecking_ball | count readback | | 0.207 (0.427) | |
| wrecking_ball | total | 0.508 (0.531) | 1.727 (2.930) | 3.4x |
| kimono | build | 0.263 (0.273) | 1.583 (2.812) | 6.0x |
| kimono | PT + EE query | 2.214 (2.259) | 26.9 (27.2) | 12.2x |
| kimono |   PT query | | 0.460 (0.518) | |
| kimono |   EE query | | 26.5 (26.8) | |
| kimono | PH query | 0.009 (0.009) | 0.209 (0.386) | 23.6x |
| kimono | narrow | 0.049 (0.053) | 0.109 (0.244) | 2.2x |
| kimono | count readback | | 0.325 (0.611) | |
| kimono | total | 2.569 (2.632) | 29.0 (30.0) | 11.3x |
| cloth_stack_x3 | build | 0.370 (0.376) | 2.588 (2.903) | 7.0x |
| cloth_stack_x3 | PT + EE query | 11.4 (11.9) | 15.1 (15.5) | 1.3x |
| cloth_stack_x3 |   PT query | | 2.550 (2.608) | |
| cloth_stack_x3 |   EE query | | 12.6 (12.9) | |
| cloth_stack_x3 | PH query | 0.000 (0.000) | 0.007 (0.011) |  |
| cloth_stack_x3 | narrow | 0.159 (0.169) | 0.242 (0.317) | 1.5x |
| cloth_stack_x3 | count readback | | 0.425 (0.503) | |
| cloth_stack_x3 | total | 12.0 (12.5) | 18.2 (18.6) | 1.5x |

## Timing, median over frames of the per-frame median (p90), ms, bvh.py inline

| scene | stage | CGQ | bvh.py | bvh.py / CGQ |
|---|---|---|---|---|
| two_bunnies | build | 0.193 (0.194) | 0.525 (0.852) | 2.7x |
| two_bunnies | PT + EE query | 0.121 (0.122) | 1.240 (1.360) | 10.2x |
| two_bunnies |   PT query | | 0.524 (0.571) | |
| two_bunnies |   EE query | | 0.713 (0.756) | |
| two_bunnies | PH query | 0.000 (0.000) | 0.003 (0.006) |  |
| two_bunnies | narrow | 0.034 (0.051) | 0.075 (0.159) | 2.2x |
| two_bunnies | count readback | | 0.211 (0.363) | |
| two_bunnies | total | 0.362 (0.381) | 1.991 (2.401) | 5.5x |
| cloth_stack | build | 0.223 (0.225) | 1.040 (2.248) | 4.7x |
| cloth_stack | PT + EE query | 0.559 (0.563) | 6.282 (6.472) | 11.2x |
| cloth_stack |   PT query | | 1.851 (1.924) | |
| cloth_stack |   EE query | | 4.393 (4.498) | |
| cloth_stack | PH query | 0.000 (0.000) | 0.004 (0.009) |  |
| cloth_stack | narrow | 0.050 (0.051) | 0.114 (0.246) | 2.3x |
| cloth_stack | count readback | | 0.245 (0.503) | |
| cloth_stack | total | 0.835 (0.842) | 7.658 (9.040) | 9.2x |
| wrecking_ball | build | 0.220 (0.223) | 0.937 (1.076) | 4.3x |
| wrecking_ball | PT + EE query | 0.210 (0.211) | 2.119 (2.146) | 10.1x |
| wrecking_ball |   PT query | | 0.815 (0.851) | |
| wrecking_ball |   EE query | | 1.304 (1.317) | |
| wrecking_ball | PH query | 0.007 (0.007) | 0.061 (0.111) | 8.6x |
| wrecking_ball | narrow | 0.049 (0.068) | 0.069 (0.097) | 1.4x |
| wrecking_ball | count readback | | 0.200 (0.265) | |
| wrecking_ball | total | 0.508 (0.531) | 3.257 (3.553) | 6.4x |
| kimono | build | 0.263 (0.273) | 1.671 (2.921) | 6.4x |
| kimono | PT + EE query | 2.214 (2.259) | 129.3 (129.8) | 58.4x |
| kimono |   PT query | | 2.596 (2.719) | |
| kimono |   EE query | | 127.0 (127.8) | |
| kimono | PH query | 0.009 (0.009) | 0.264 (0.436) | 29.7x |
| kimono | narrow | 0.049 (0.053) | 0.126 (0.255) | 2.6x |
| kimono | count readback | | 0.385 (0.670) | |
| kimono | total | 2.569 (2.632) | 131.7 (132.6) | 51.3x |
| cloth_stack_x3 | build | 0.370 (0.376) | 2.461 (3.131) | 6.6x |
| cloth_stack_x3 | PT + EE query | 11.4 (11.9) | 46.2 (47.4) | 4.0x |
| cloth_stack_x3 |   PT query | | 8.053 (8.297) | |
| cloth_stack_x3 |   EE query | | 38.2 (39.3) | |
| cloth_stack_x3 | PH query | 0.000 (0.000) | 0.006 (0.012) |  |
| cloth_stack_x3 | narrow | 0.159 (0.169) | 0.272 (0.361) | 1.7x |
| cloth_stack_x3 | count readback | | 0.343 (0.598) | |
| cloth_stack_x3 | total | 12.0 (12.5) | 49.3 (50.4) | 4.1x |
