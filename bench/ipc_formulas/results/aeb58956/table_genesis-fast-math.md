# IPC formula parity: genesis-fast-math

Per case and input set: rows, then per backend the failing rows, the largest real error over the scale of its output (max |g - c| / s), and the rows that passed by the order clause and by the referee (compare.py).

| case | function | set | rows | cuda_fast-math-true failures | cuda_fast-math-true max err / s | cuda_fast-math-true order rows | cuda_fast-math-true referee rows | cpu_fast-math-true failures | cpu_fast-math-true max err / s | cpu_fast-math-true order rows | cpu_fast-math-true referee rows |
|---|---|---|---|---|---|---|---|---|---|---|---|
| barrier | barrier, derivatives, normal force | degenerate | 4000 | 0 | 3.32e-16 | 0 | 0 | 0 | 4.31e-05 | 0 | 0 |
| barrier | barrier, derivatives, normal force | random | 4000 | 0 | 4.12e-16 | 0 | 0 | 0 | 1.12e-15 | 0 | 0 |
| barrier_ee | EE barrier gradient/Hessian (mollifier and flag dispatch) | degenerate | 5280 | 215 | 4.48e-10 | 0 | 0 | 0 | 3.90e-12 | 0 | 0 |
| barrier_ee | EE barrier gradient/Hessian (mollifier and flag dispatch) | mollifier_sweep | 4000 | 526 | 8.84e-02 | 0 | 0 | 0 | 1.71e-13 | 0 | 0 |
| barrier_ee | EE barrier gradient/Hessian (mollifier and flag dispatch) | parallel | 576 | 3 | 1.08e-15 | 0 | 0 | 0 | 4.67e-02 | 0 | 0 |
| barrier_ee | EE barrier gradient/Hessian (mollifier and flag dispatch) | random | 4000 | 0 | 1.14e-13 | 0 | 0 | 0 | 3.20e-13 | 0 | 0 |
| barrier_ee | EE barrier gradient/Hessian (mollifier and flag dispatch) | real_cloth_stack | 30000 | 12 | 7.62e-15 | 0 | 0 | 0 | 7.41e-15 | 0 | 0 |
| barrier_ee | EE barrier gradient/Hessian (mollifier and flag dispatch) | real_cloth_stack_mollified | 18457 | 0 | 1.44e-14 | 0 | 0 | 0 | 2.29e-14 | 0 | 0 |
| barrier_ee | EE barrier gradient/Hessian (mollifier and flag dispatch) | real_kimono | 30000 | 0 | 2.02e-16 | 0 | 0 | 0 | 2.02e-16 | 0 | 0 |
| barrier_ee | EE barrier gradient/Hessian (mollifier and flag dispatch) | real_kimono_mollified | 514 | 0 | 3.17e-15 | 0 | 0 | 0 | 4.93e-15 | 0 | 0 |
| barrier_pe | PE barrier gradient/Hessian (flag dispatch) | degenerate | 256 | 2 | 2.22e-12 | 0 | 0 | 0 | 2.17e-11 | 0 | 0 |
| barrier_pe | PE barrier gradient/Hessian (flag dispatch) | random | 4000 | 0 | 3.27e-15 | 0 | 0 | 0 | 3.21e-15 | 0 | 0 |
| barrier_pe | PE barrier gradient/Hessian (flag dispatch) | real_cloth_stack | 17883 | 0 | 1.28e-14 | 0 | 0 | 0 | 1.84e-14 | 0 | 0 |
| barrier_pe | PE barrier gradient/Hessian (flag dispatch) | real_kimono | 12741 | 0 | 4.29e-16 | 0 | 0 | 0 | 4.29e-16 | 0 | 0 |
| barrier_ph | PH barrier gradient/Hessian | degenerate | 1476 | 0 | 1.71e-16 | 0 | 0 | 0 | 3.03e-16 | 0 | 0 |
| barrier_ph | PH barrier gradient/Hessian | random | 4000 | 0 | 7.75e-15 | 0 | 0 | 0 | 2.21e-14 | 0 | 0 |
| barrier_pp | PP barrier gradient/Hessian | degenerate | 64 | 0 | 4.26e-16 | 0 | 0 | 0 | 1.08e-11 | 0 | 0 |
| barrier_pp | PP barrier gradient/Hessian | random | 4000 | 0 | 1.89e-15 | 0 | 0 | 0 | 4.19e-16 | 0 | 0 |
| barrier_pp | PP barrier gradient/Hessian | real_cloth_stack | 1404 | 0 | 8.67e-15 | 0 | 0 | 0 | 7.41e-15 | 0 | 0 |
| barrier_pp | PP barrier gradient/Hessian | real_kimono | 882 | 0 | 2.02e-16 | 0 | 0 | 0 | 2.02e-16 | 0 | 0 |
| barrier_pt | PT barrier gradient/Hessian (flag dispatch) | degenerate | 2048 | 11 | 3.26e-11 | 0 | 0 | 0 | 1.27e-11 | 0 | 0 |
| barrier_pt | PT barrier gradient/Hessian (flag dispatch) | random | 4000 | 0 | 3.94e-16 | 0 | 0 | 0 | 1.47e-13 | 0 | 0 |
| barrier_pt | PT barrier gradient/Hessian (flag dispatch) | real_cloth_stack | 30000 | 112 | 2.68e-15 | 0 | 0 | 0 | 5.09e-15 | 0 | 0 |
| barrier_pt | PT barrier gradient/Hessian (flag dispatch) | real_kimono | 30000 | 0 | 3.01e-16 | 0 | 0 | 0 | 6.01e-16 | 0 | 0 |
| candidate_ee | EE broad-phase distance rule | degenerate | 1280 | 0 | - | 0 | 0 | 0 | - | 0 | 0 |
| candidate_ee | EE broad-phase distance rule | random | 4000 | 0 | - | 0 | 0 | 0 | - | 0 | 0 |
| candidate_ee | EE broad-phase distance rule | real_cloth_stack | 30000 | 0 | - | 0 | 0 | 0 | - | 0 | 0 |
| candidate_ee | EE broad-phase distance rule | real_kimono | 30000 | 0 | - | 0 | 0 | 0 | - | 0 | 0 |
| candidate_pe | PE broad-phase distance rule | degenerate | 256 | 0 | - | 0 | 0 | 0 | - | 0 | 0 |
| candidate_pe | PE broad-phase distance rule | random | 4000 | 0 | - | 0 | 0 | 0 | - | 0 | 0 |
| candidate_ph | PH broad-phase cull | degenerate | 1476 | 0 | - | 0 | 0 | 0 | - | 0 | 0 |
| candidate_ph | PH broad-phase cull | random | 4000 | 0 | - | 0 | 0 | 0 | - | 0 | 0 |
| candidate_pp | PP broad-phase distance rule | degenerate | 64 | 0 | - | 0 | 0 | 0 | - | 0 | 0 |
| candidate_pp | PP broad-phase distance rule | random | 4000 | 0 | - | 0 | 0 | 0 | - | 0 | 0 |
| candidate_pt | PT broad-phase distance rule | degenerate | 2048 | 0 | - | 0 | 0 | 0 | - | 0 | 0 |
| candidate_pt | PT broad-phase distance rule | random | 4000 | 0 | - | 0 | 0 | 0 | - | 0 | 0 |
| candidate_pt | PT broad-phase distance rule | real_cloth_stack | 30000 | 0 | - | 0 | 0 | 0 | - | 0 | 0 |
| candidate_pt | PT broad-phase distance rule | real_kimono | 30000 | 0 | - | 0 | 0 | 0 | - | 0 | 0 |
| ccd_additive_ee | EE additive CCD | degenerate | 4560 | 17 | 8.26e-11 | 0 | 0 | 3 | 1.64e-10 | 0 | 0 |
| ccd_additive_ee | EE additive CCD | random | 4000 | 0 | 8.55e-14 | 0 | 0 | 0 | 1.20e-13 | 0 | 0 |
| ccd_additive_ee | EE additive CCD | real_cloth_stack | 30000 | 0 | 0.00e+00 | 0 | 0 | 0 | 0.00e+00 | 0 | 0 |
| ccd_additive_ee | EE additive CCD | real_kimono | 30000 | 0 | 2.89e-15 | 0 | 0 | 0 | 1.55e-15 | 0 | 0 |
| ccd_additive_pe | PE additive CCD | degenerate | 512 | 0 | 7.34e-13 | 0 | 0 | 0 | 1.55e-12 | 0 | 0 |
| ccd_additive_pe | PE additive CCD | random | 4000 | 0 | 1.78e-15 | 0 | 0 | 0 | 6.66e-16 | 0 | 0 |
| ccd_additive_pp | PP additive CCD | degenerate | 128 | 0 | 4.59e-14 | 0 | 0 | 0 | 2.13e-14 | 0 | 0 |
| ccd_additive_pp | PP additive CCD | random | 4000 | 0 | 2.22e-16 | 0 | 0 | 0 | 2.78e-16 | 0 | 0 |
| ccd_additive_pt | PT additive CCD | degenerate | 4096 | 3 | 3.51e-12 | 0 | 0 | 2 | 3.91e-12 | 0 | 0 |
| ccd_additive_pt | PT additive CCD | random | 4000 | 0 | 7.03e-14 | 0 | 0 | 0 | 7.18e-14 | 0 | 0 |
| ccd_additive_pt | PT additive CCD | real_cloth_stack | 30000 | 0 | 0.00e+00 | 0 | 0 | 0 | 0.00e+00 | 0 | 0 |
| ccd_additive_pt | PT additive CCD | real_kimono | 30000 | 0 | 5.77e-15 | 0 | 0 | 0 | 5.88e-15 | 0 | 0 |
| ccd_directional_ee | EE directional CCD | degenerate | 4560 | 127 | 9.36e-08 | 0 | 0 | 25 | 1.15e-07 | 0 | 0 |
| ccd_directional_ee | EE directional CCD | random | 4000 | 0 | 2.99e-11 | 0 | 0 | 1 | 2.63e-11 | 0 | 0 |
| ccd_directional_ee | EE directional CCD | real_cloth_stack | 30000 | 0 | 0.00e+00 | 0 | 0 | 0 | 0.00e+00 | 0 | 0 |
| ccd_directional_ee | EE directional CCD | real_kimono | 30000 | 0 | 6.76e-12 | 0 | 0 | 0 | 7.37e-12 | 0 | 0 |
| ccd_directional_pe | PE directional CCD | degenerate | 512 | 1 | 2.01e-12 | 0 | 0 | 1 | 1.58e-12 | 0 | 0 |
| ccd_directional_pe | PE directional CCD | random | 4000 | 0 | 4.69e-14 | 0 | 0 | 0 | 3.29e-14 | 0 | 0 |
| ccd_directional_pp | PP directional CCD | degenerate | 128 | 0 | 1.32e-13 | 0 | 0 | 0 | 6.84e-14 | 0 | 0 |
| ccd_directional_pp | PP directional CCD | random | 4000 | 0 | 4.44e-16 | 0 | 0 | 0 | 8.33e-16 | 0 | 0 |
| ccd_directional_pt | PT directional CCD | degenerate | 4096 | 3 | 2.89e-12 | 0 | 0 | 2 | 2.08e-12 | 0 | 0 |
| ccd_directional_pt | PT directional CCD | random | 4000 | 0 | 5.83e-12 | 0 | 0 | 0 | 3.91e-11 | 0 | 0 |
| ccd_directional_pt | PT directional CCD | real_cloth_stack | 30000 | 0 | 0.00e+00 | 0 | 0 | 0 | 0.00e+00 | 0 | 0 |
| ccd_directional_pt | PT directional CCD | real_kimono | 30000 | 53 | 3.10e-01 | 0 | 0 | 2 | 3.20e-01 | 0 | 0 |
| ccd_ph | PH CCD (linear) | degenerate | 1476 | 0 | 4.11e-17 | 0 | 0 | 0 | 4.11e-17 | 0 | 0 |
| ccd_ph | PH CCD (linear) | random | 4000 | 0 | 2.30e-14 | 0 | 0 | 0 | 3.99e-14 | 0 | 0 |
| ccd_ph_screw | PH screw CCD | degenerate | 1476 | 0 | 2.11e-15 | 0 | 0 | 0 | 2.00e-15 | 0 | 0 |
| ccd_ph_screw | PH screw CCD | random | 4000 | 0 | 2.78e-14 | 0 | 0 | 0 | 3.83e-14 | 0 | 0 |
| ccd_screw_ee | EE screw CCD (whole step) | degenerate | 1280 | 4 | 1.17e-01 | 0 | 0 | 3 | 1.93e-01 | 0 | 0 |
| ccd_screw_ee | EE screw CCD (whole step) | random | 4000 | 12 | 4.02e-01 | 0 | 0 | 11 | 4.02e-01 | 0 | 0 |
| ccd_screw_interval_ee | EE screw CCD (interval) | degenerate | 1280 | 2 | 4.75e-02 | 0 | 0 | 1 | 2.97e-02 | 0 | 0 |
| ccd_screw_interval_ee | EE screw CCD (interval) | random | 4000 | 4 | 2.83e-01 | 0 | 0 | 3 | 2.93e-01 | 0 | 0 |
| ccd_screw_interval_pt | PT screw CCD (interval) | degenerate | 2048 | 0 | 1.94e-14 | 0 | 0 | 0 | 1.67e-14 | 0 | 0 |
| ccd_screw_interval_pt | PT screw CCD (interval) | random | 4000 | 0 | 2.55e-01 | 0 | 0 | 1 | 2.69e-01 | 0 | 0 |
| ccd_screw_pe | PE screw CCD (whole step) | degenerate | 256 | 0 | 1.44e-15 | 0 | 0 | 0 | 1.05e-15 | 0 | 0 |
| ccd_screw_pe | PE screw CCD (whole step) | random | 4000 | 0 | 1.72e-14 | 0 | 0 | 0 | 5.87e-14 | 0 | 0 |
| ccd_screw_pp | PP screw CCD (whole step) | degenerate | 64 | 0 | 3.33e-16 | 0 | 0 | 0 | 2.22e-16 | 0 | 0 |
| ccd_screw_pp | PP screw CCD (whole step) | random | 4000 | 0 | 8.88e-16 | 0 | 0 | 0 | 9.99e-16 | 0 | 0 |
| ccd_screw_pt | PT screw CCD (whole step) | degenerate | 2048 | 7 | 3.13e-01 | 0 | 0 | 8 | 3.13e-01 | 0 | 0 |
| ccd_screw_pt | PT screw CCD (whole step) | random | 4000 | 1 | 3.23e-01 | 0 | 0 | 3 | 2.69e-01 | 0 | 0 |
| closest_ee | EE closest points | degenerate | 3632 | 245 | 1.61e-01 | 0 | 0 | 179 | 1.55e-01 | 0 | 0 |
| closest_ee | EE closest points | random | 4000 | 0 | 8.86e-14 | 0 | 0 | 0 | 6.67e-14 | 0 | 0 |
| closest_ee | EE closest points | real_cloth_stack | 30000 | 0 | 4.17e-15 | 0 | 0 | 0 | 4.17e-15 | 0 | 0 |
| closest_ee | EE closest points | real_kimono | 30000 | 0 | 8.05e-16 | 0 | 0 | 0 | 1.02e-15 | 0 | 0 |
| closest_pe | PE closest point | degenerate | 336 | 0 | 0.00e+00 | 0 | 0 | 0 | 0.00e+00 | 0 | 0 |
| closest_pe | PE closest point | random | 4000 | 0 | 5.55e-16 | 0 | 0 | 0 | 5.55e-16 | 0 | 0 |
| closest_pt | PT closest point | degenerate | 2580 | 0 | 0.00e+00 | 0 | 0 | 0 | 0.00e+00 | 0 | 0 |
| closest_pt | PT closest point | random | 4000 | 0 | 6.67e-16 | 0 | 0 | 0 | 1.11e-15 | 0 | 0 |
| closest_pt | PT closest point | real_cloth_stack | 30000 | 0 | 4.28e-15 | 0 | 0 | 0 | 4.28e-15 | 0 | 0 |
| closest_pt | PT closest point | real_kimono | 30000 | 0 | 3.82e-16 | 0 | 0 | 0 | 3.87e-16 | 0 | 0 |
| ee_distance | EE distance flag, flagged distance2 and gradient | degenerate | 3632 | 106 | 8.59e-13 | 0 | 0 | 0 | 3.13e-13 | 0 | 0 |
| ee_distance | EE distance flag, flagged distance2 and gradient | random | 4000 | 0 | 1.29e-15 | 0 | 0 | 0 | 2.64e-15 | 0 | 0 |
| ee_distance | EE distance flag, flagged distance2 and gradient | real_cloth_stack | 30000 | 4 | 3.79e-15 | 0 | 0 | 0 | 1.90e-15 | 0 | 0 |
| ee_distance | EE distance flag, flagged distance2 and gradient | real_kimono | 30000 | 0 | 2.62e-16 | 0 | 0 | 0 | 2.83e-16 | 0 | 0 |
| ee_distance_type | EE distance type (parallel projection) | degenerate | 3632 | 210 | - | 0 | 0 | 140 | - | 0 | 0 |
| ee_distance_type | EE distance type (parallel projection) | random | 4000 | 0 | - | 0 | 0 | 0 | - | 0 | 0 |
| ee_distance_type | EE distance type (parallel projection) | real_cloth_stack | 30000 | 0 | - | 0 | 0 | 0 | - | 0 | 0 |
| ee_distance_type | EE distance type (parallel projection) | real_kimono | 30000 | 0 | - | 0 | 0 | 0 | - | 0 | 0 |
| frame_ee | EE tangent basis and closest-point weights | degenerate | 868 | 0 | 1.38e-13 | 0 | 0 | 0 | 2.03e-13 | 0 | 0 |
| frame_ee | EE tangent basis and closest-point weights | random | 3998 | 0 | 8.42e-14 | 0 | 0 | 0 | 2.48e-13 | 0 | 0 |
| frame_ee | EE tangent basis and closest-point weights | real_cloth_stack | 19905 | 0 | 4.68e-15 | 0 | 0 | 0 | 5.46e-15 | 0 | 0 |
| frame_ee | EE tangent basis and closest-point weights | real_kimono | 19902 | 0 | 3.17e-13 | 0 | 0 | 0 | 4.60e-13 | 0 | 0 |
| frame_pe | PE tangent basis and closest-point weights | degenerate | 256 | 0 | 0.00e+00 | 0 | 0 | 0 | 0.00e+00 | 0 | 0 |
| frame_pe | PE tangent basis and closest-point weights | random | 4000 | 0 | 3.33e-16 | 0 | 0 | 0 | 1.11e-15 | 0 | 0 |
| frame_pe | PE tangent basis and closest-point weights | real_cloth_stack | 17883 | 0 | 3.33e-16 | 0 | 0 | 0 | 2.55e-14 | 0 | 0 |
| frame_pe | PE tangent basis and closest-point weights | real_kimono | 12741 | 0 | 4.44e-16 | 0 | 0 | 0 | 1.65e-14 | 0 | 0 |
| frame_ph | PH tangent basis (projector) | degenerate | 1476 | 0 | 0.00e+00 | 0 | 0 | 0 | 0.00e+00 | 0 | 0 |
| frame_ph | PH tangent basis (projector) | random | 4000 | 0 | 7.77e-16 | 0 | 0 | 0 | 8.88e-16 | 0 | 0 |
| frame_pp | PP tangent basis (projector) | degenerate | 64 | 0 | 4.44e-16 | 0 | 0 | 0 | 4.44e-16 | 0 | 0 |
| frame_pp | PP tangent basis (projector) | random | 4000 | 0 | 8.88e-16 | 0 | 0 | 0 | 9.99e-16 | 0 | 0 |
| frame_pp | PP tangent basis (projector) | real_cloth_stack | 1404 | 0 | 8.88e-16 | 0 | 0 | 0 | 8.88e-16 | 0 | 0 |
| frame_pp | PP tangent basis (projector) | real_kimono | 882 | 0 | 8.88e-16 | 0 | 0 | 0 | 8.88e-16 | 0 | 0 |
| frame_pt | PT tangent basis and closest-point weights | degenerate | 2048 | 0 | 0.00e+00 | 0 | 0 | 0 | 0.00e+00 | 0 | 0 |
| frame_pt | PT tangent basis and closest-point weights | random | 4000 | 0 | 6.97e-14 | 0 | 0 | 0 | 1.61e-13 | 0 | 0 |
| frame_pt | PT tangent basis and closest-point weights | real_cloth_stack | 20807 | 0 | 1.22e-15 | 0 | 0 | 0 | 1.22e-15 | 0 | 0 |
| frame_pt | PT tangent basis and closest-point weights | real_kimono | 26473 | 0 | 1.78e-15 | 0 | 0 | 0 | 1.68e-15 | 0 | 0 |
| friction_ee | EE friction energy/gradient/Hessian | degenerate | 868 | 0 | 3.50e-14 | 0 | 0 | 0 | 6.11e-14 | 0 | 0 |
| friction_ee | EE friction energy/gradient/Hessian | random | 3998 | 0 | 3.22e-14 | 0 | 0 | 0 | 9.67e-14 | 0 | 0 |
| friction_ee | EE friction energy/gradient/Hessian | real_cloth_stack | 19905 | 0 | 8.36e-15 | 0 | 0 | 0 | 9.36e-15 | 0 | 0 |
| friction_ee | EE friction energy/gradient/Hessian | real_kimono | 19902 | 0 | 1.01e-14 | 0 | 0 | 0 | 6.59e-15 | 0 | 0 |
| friction_pe | PE friction energy/gradient/Hessian | degenerate | 256 | 0 | 3.98e-16 | 0 | 0 | 0 | 1.99e-16 | 0 | 0 |
| friction_pe | PE friction energy/gradient/Hessian | random | 4000 | 0 | 1.55e-15 | 0 | 0 | 0 | 2.57e-15 | 0 | 0 |
| friction_pe | PE friction energy/gradient/Hessian | real_cloth_stack | 17883 | 0 | 1.27e-14 | 0 | 0 | 0 | 1.30e-14 | 0 | 0 |
| friction_pe | PE friction energy/gradient/Hessian | real_kimono | 12741 | 0 | 1.07e-14 | 0 | 0 | 0 | 8.42e-15 | 0 | 0 |
| friction_ph | PH friction energy/gradient/Hessian | degenerate | 1476 | 0 | 4.09e-16 | 0 | 0 | 0 | 4.09e-16 | 0 | 0 |
| friction_ph | PH friction energy/gradient/Hessian | random | 4000 | 0 | 7.14e-14 | 0 | 0 | 0 | 2.13e-13 | 0 | 0 |
| friction_pp | PP friction energy/gradient/Hessian | degenerate | 64 | 0 | 6.32e-16 | 0 | 0 | 0 | 3.06e-16 | 0 | 0 |
| friction_pp | PP friction energy/gradient/Hessian | random | 4000 | 0 | 1.67e-15 | 0 | 0 | 0 | 1.67e-15 | 0 | 0 |
| friction_pp | PP friction energy/gradient/Hessian | real_cloth_stack | 1404 | 0 | 1.42e-14 | 0 | 0 | 0 | 1.12e-14 | 0 | 0 |
| friction_pp | PP friction energy/gradient/Hessian | real_kimono | 882 | 0 | 7.67e-16 | 0 | 0 | 0 | 2.30e-16 | 0 | 0 |
| friction_pt | PT friction energy/gradient/Hessian | degenerate | 2048 | 2 | 2.86e-12 | 0 | 0 | 2 | 2.86e-12 | 0 | 0 |
| friction_pt | PT friction energy/gradient/Hessian | random | 4000 | 3 | 2.40e-12 | 0 | 0 | 3 | 2.40e-12 | 0 | 0 |
| friction_pt | PT friction energy/gradient/Hessian | real_cloth_stack | 20807 | 0 | 1.67e-13 | 0 | 0 | 0 | 1.67e-13 | 0 | 0 |
| friction_pt | PT friction energy/gradient/Hessian | real_kimono | 26473 | 1 | 3.77e-12 | 0 | 0 | 1 | 3.77e-12 | 0 | 0 |
| intersect | edge-triangle intersection | degenerate | 800 | 0 | - | 0 | 0 | 0 | - | 0 | 0 |
| intersect | edge-triangle intersection | random | 4000 | 0 | - | 0 | 0 | 0 | - | 0 | 0 |
| mollified_barrier | mollified barrier | degenerate | 4000 | 0 | 2.95e-16 | 0 | 0 | 0 | 7.83e-07 | 0 | 0 |
| mollified_barrier | mollified barrier | random | 4000 | 0 | 3.63e-16 | 0 | 0 | 0 | 3.63e-16 | 0 | 0 |
| mollifier | mollifier threshold and edge cross norm2 | degenerate | 3632 | 0 | 2.27e-16 | 0 | 0 | 0 | 2.27e-16 | 0 | 0 |
| mollifier | mollifier threshold and edge cross norm2 | random | 4000 | 0 | 2.06e-16 | 0 | 0 | 0 | 2.38e-16 | 0 | 0 |
| mollifier | mollifier threshold and edge cross norm2 | real_cloth_stack | 30000 | 0 | 1.62e-20 | 0 | 0 | 0 | 2.48e-20 | 0 | 0 |
| mollifier | mollifier threshold and edge cross norm2 | real_kimono | 30000 | 0 | 1.61e-16 | 0 | 0 | 0 | 4.84e-16 | 0 | 0 |
| pair_ee | EE pair d_hat, thickness, area weight | random | 4000 | 0 | 0.00e+00 | 0 | 0 | 0 | 0.00e+00 | 0 | 0 |
| pair_pe | PE pair d_hat, thickness, area weight | random | 4000 | 0 | 0.00e+00 | 0 | 0 | 0 | 0.00e+00 | 0 | 0 |
| pair_pp | PP pair d_hat, thickness, area weight | random | 4000 | 0 | 0.00e+00 | 0 | 0 | 0 | 0.00e+00 | 0 | 0 |
| pair_pt | PT pair d_hat, thickness, area weight | random | 4000 | 0 | 0.00e+00 | 0 | 0 | 0 | 0.00e+00 | 0 | 0 |
| pe_distance | PE distance flag, flagged distance2 and gradient | degenerate | 336 | 0 | 0.00e+00 | 0 | 0 | 0 | 0.00e+00 | 0 | 0 |
| pe_distance | PE distance flag, flagged distance2 and gradient | random | 4000 | 0 | 3.90e-15 | 0 | 0 | 0 | 4.81e-15 | 0 | 0 |
| pe_distance | PE distance flag, flagged distance2 and gradient | real_cloth_stack | 17883 | 0 | 4.16e-15 | 0 | 0 | 0 | 2.08e-15 | 0 | 0 |
| pe_distance | PE distance flag, flagged distance2 and gradient | real_kimono | 12741 | 0 | 1.22e-14 | 0 | 0 | 0 | 7.20e-15 | 0 | 0 |
| pt_distance | PT distance flag, flagged distance2 and gradient | degenerate | 2580 | 0 | 0.00e+00 | 0 | 0 | 0 | 0.00e+00 | 0 | 0 |
| pt_distance | PT distance flag, flagged distance2 and gradient | random | 4000 | 0 | 1.24e-15 | 0 | 0 | 0 | 1.95e-15 | 0 | 0 |
| pt_distance | PT distance flag, flagged distance2 and gradient | real_cloth_stack | 30000 | 75 | 3.84e-15 | 0 | 0 | 0 | 1.92e-15 | 0 | 0 |
| pt_distance | PT distance flag, flagged distance2 and gradient | real_kimono | 30000 | 0 | 2.15e-16 | 0 | 0 | 0 | 2.15e-16 | 0 | 0 |

Totals: cuda_fast-math-true: 1761 failing rows of 1452423, cpu_fast-math-true: 391 failing rows of 1452423
