# IPC formula parity: acceptance-noreferee

Per case and input set: rows, then per backend the failing rows, the largest real error over the scale of its output (max |g - c| / s), and the rows that passed by the order clause and by the referee (compare.py).

| case | function | set | rows | cuda_fast-math-false failures | cuda_fast-math-false max err / s | cuda_fast-math-false order rows | cuda_fast-math-false referee rows | cpu_fast-math-false failures | cpu_fast-math-false max err / s | cpu_fast-math-false order rows | cpu_fast-math-false referee rows |
|---|---|---|---|---|---|---|---|---|---|---|---|
| barrier | barrier, derivatives, normal force | degenerate | 4000 | 0 | 4.31e-05 | 0 | 0 | 0 | 4.31e-05 | 0 | 0 |
| barrier | barrier, derivatives, normal force | random | 4000 | 0 | 1.03e-15 | 0 | 0 | 0 | 1.03e-15 | 0 | 0 |
| barrier_ee | EE barrier gradient/Hessian (mollifier and flag dispatch) | degenerate | 5280 | 0 | 3.90e-12 | 0 | 0 | 0 | 3.90e-12 | 0 | 0 |
| barrier_ee | EE barrier gradient/Hessian (mollifier and flag dispatch) | mollifier_sweep | 4000 | 0 | 1.71e-13 | 0 | 0 | 0 | 1.71e-13 | 0 | 0 |
| barrier_ee | EE barrier gradient/Hessian (mollifier and flag dispatch) | parallel | 576 | 0 | 4.67e-02 | 0 | 0 | 0 | 4.67e-02 | 0 | 0 |
| barrier_ee | EE barrier gradient/Hessian (mollifier and flag dispatch) | random | 4000 | 0 | 2.05e-13 | 0 | 0 | 0 | 2.05e-13 | 0 | 0 |
| barrier_ee | EE barrier gradient/Hessian (mollifier and flag dispatch) | real_cloth_stack | 30000 | 0 | 7.41e-15 | 0 | 0 | 0 | 7.41e-15 | 0 | 0 |
| barrier_ee | EE barrier gradient/Hessian (mollifier and flag dispatch) | real_cloth_stack_mollified | 18457 | 0 | 2.01e-14 | 0 | 0 | 0 | 2.01e-14 | 0 | 0 |
| barrier_ee | EE barrier gradient/Hessian (mollifier and flag dispatch) | real_kimono | 30000 | 0 | 2.02e-16 | 0 | 0 | 0 | 2.02e-16 | 0 | 0 |
| barrier_ee | EE barrier gradient/Hessian (mollifier and flag dispatch) | real_kimono_mollified | 514 | 0 | 2.47e-15 | 0 | 0 | 0 | 2.47e-15 | 0 | 0 |
| barrier_pe | PE barrier gradient/Hessian (flag dispatch) | degenerate | 256 | 0 | 2.17e-11 | 0 | 0 | 0 | 2.17e-11 | 0 | 0 |
| barrier_pe | PE barrier gradient/Hessian (flag dispatch) | random | 4000 | 0 | 5.25e-15 | 0 | 0 | 0 | 5.25e-15 | 0 | 0 |
| barrier_pe | PE barrier gradient/Hessian (flag dispatch) | real_cloth_stack | 17883 | 0 | 1.84e-14 | 0 | 0 | 0 | 1.84e-14 | 0 | 0 |
| barrier_pe | PE barrier gradient/Hessian (flag dispatch) | real_kimono | 12741 | 0 | 3.75e-16 | 0 | 0 | 0 | 3.75e-16 | 0 | 0 |
| barrier_ph | PH barrier gradient/Hessian | degenerate | 1476 | 0 | 1.90e-16 | 0 | 0 | 0 | 1.90e-16 | 0 | 0 |
| barrier_ph | PH barrier gradient/Hessian | random | 4000 | 0 | 2.21e-14 | 0 | 0 | 0 | 2.21e-14 | 0 | 0 |
| barrier_pp | PP barrier gradient/Hessian | degenerate | 64 | 0 | 1.08e-11 | 0 | 0 | 0 | 1.08e-11 | 0 | 0 |
| barrier_pp | PP barrier gradient/Hessian | random | 4000 | 0 | 4.19e-16 | 0 | 0 | 0 | 4.19e-16 | 0 | 0 |
| barrier_pp | PP barrier gradient/Hessian | real_cloth_stack | 1404 | 0 | 7.41e-15 | 0 | 0 | 0 | 7.41e-15 | 0 | 0 |
| barrier_pp | PP barrier gradient/Hessian | real_kimono | 882 | 0 | 2.02e-16 | 0 | 0 | 0 | 2.02e-16 | 0 | 0 |
| barrier_pt | PT barrier gradient/Hessian (flag dispatch) | degenerate | 2048 | 0 | 1.27e-11 | 0 | 0 | 0 | 1.27e-11 | 0 | 0 |
| barrier_pt | PT barrier gradient/Hessian (flag dispatch) | random | 4000 | 0 | 9.15e-14 | 0 | 0 | 0 | 9.15e-14 | 0 | 0 |
| barrier_pt | PT barrier gradient/Hessian (flag dispatch) | real_cloth_stack | 30000 | 0 | 5.09e-15 | 0 | 0 | 0 | 5.09e-15 | 0 | 0 |
| barrier_pt | PT barrier gradient/Hessian (flag dispatch) | real_kimono | 30000 | 0 | 4.53e-16 | 0 | 0 | 0 | 4.53e-16 | 0 | 0 |
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
| ccd_additive_ee | EE additive CCD | degenerate | 4560 | 0 | 1.64e-10 | 19 | 0 | 0 | 1.64e-10 | 19 | 0 |
| ccd_additive_ee | EE additive CCD | random | 4000 | 0 | 1.20e-13 | 0 | 0 | 0 | 1.20e-13 | 0 | 0 |
| ccd_additive_ee | EE additive CCD | real_cloth_stack | 30000 | 0 | 0.00e+00 | 0 | 0 | 0 | 0.00e+00 | 0 | 0 |
| ccd_additive_ee | EE additive CCD | real_kimono | 30000 | 0 | 1.55e-15 | 0 | 0 | 0 | 1.55e-15 | 0 | 0 |
| ccd_additive_pe | PE additive CCD | degenerate | 512 | 0 | 1.23e-12 | 0 | 0 | 0 | 1.23e-12 | 0 | 0 |
| ccd_additive_pe | PE additive CCD | random | 4000 | 0 | 5.55e-16 | 0 | 0 | 0 | 5.55e-16 | 0 | 0 |
| ccd_additive_pp | PP additive CCD | degenerate | 128 | 0 | 4.59e-14 | 0 | 0 | 0 | 4.59e-14 | 0 | 0 |
| ccd_additive_pp | PP additive CCD | random | 4000 | 0 | 2.78e-16 | 0 | 0 | 0 | 2.78e-16 | 0 | 0 |
| ccd_additive_pt | PT additive CCD | degenerate | 4096 | 0 | 3.91e-12 | 0 | 0 | 0 | 3.91e-12 | 0 | 0 |
| ccd_additive_pt | PT additive CCD | random | 4000 | 0 | 1.10e-13 | 0 | 0 | 0 | 1.10e-13 | 0 | 0 |
| ccd_additive_pt | PT additive CCD | real_cloth_stack | 30000 | 0 | 0.00e+00 | 0 | 0 | 0 | 0.00e+00 | 0 | 0 |
| ccd_additive_pt | PT additive CCD | real_kimono | 30000 | 0 | 5.88e-15 | 0 | 0 | 0 | 5.88e-15 | 0 | 0 |
| ccd_directional_ee | EE directional CCD | degenerate | 4560 | 0 | 9.88e-08 | 99 | 0 | 0 | 9.88e-08 | 99 | 0 |
| ccd_directional_ee | EE directional CCD | random | 4000 | 0 | 4.28e-11 | 0 | 0 | 0 | 4.28e-11 | 0 | 0 |
| ccd_directional_ee | EE directional CCD | real_cloth_stack | 30000 | 0 | 0.00e+00 | 0 | 0 | 0 | 0.00e+00 | 0 | 0 |
| ccd_directional_ee | EE directional CCD | real_kimono | 30000 | 0 | 7.37e-12 | 1 | 0 | 0 | 7.37e-12 | 1 | 0 |
| ccd_directional_pe | PE directional CCD | degenerate | 512 | 0 | 1.58e-12 | 0 | 0 | 0 | 1.58e-12 | 0 | 0 |
| ccd_directional_pe | PE directional CCD | random | 4000 | 0 | 9.55e-15 | 0 | 0 | 0 | 9.55e-15 | 0 | 0 |
| ccd_directional_pp | PP directional CCD | degenerate | 128 | 0 | 6.97e-14 | 0 | 0 | 0 | 6.97e-14 | 0 | 0 |
| ccd_directional_pp | PP directional CCD | random | 4000 | 0 | 8.33e-16 | 0 | 0 | 0 | 8.33e-16 | 0 | 0 |
| ccd_directional_pt | PT directional CCD | degenerate | 4096 | 0 | 1.88e-12 | 1 | 0 | 0 | 1.88e-12 | 1 | 0 |
| ccd_directional_pt | PT directional CCD | random | 4000 | 0 | 6.28e-11 | 0 | 0 | 0 | 6.28e-11 | 0 | 0 |
| ccd_directional_pt | PT directional CCD | real_cloth_stack | 30000 | 0 | 0.00e+00 | 0 | 0 | 0 | 0.00e+00 | 0 | 0 |
| ccd_directional_pt | PT directional CCD | real_kimono | 30000 | 0 | 3.20e-01 | 0 | 0 | 0 | 3.20e-01 | 0 | 0 |
| ccd_ph | PH CCD (linear) | degenerate | 1476 | 0 | 0.00e+00 | 0 | 0 | 0 | 0.00e+00 | 0 | 0 |
| ccd_ph | PH CCD (linear) | random | 4000 | 0 | 3.99e-14 | 0 | 0 | 0 | 3.99e-14 | 0 | 0 |
| ccd_ph_screw | PH screw CCD | degenerate | 1476 | 0 | 9.99e-16 | 0 | 0 | 0 | 9.99e-16 | 0 | 0 |
| ccd_ph_screw | PH screw CCD | random | 4000 | 0 | 2.28e-14 | 0 | 0 | 0 | 2.28e-14 | 0 | 0 |
| ccd_screw_ee | EE screw CCD (whole step) | degenerate | 1280 | 0 | 1.93e-01 | 0 | 0 | 0 | 1.93e-01 | 0 | 0 |
| ccd_screw_ee | EE screw CCD (whole step) | random | 4000 | 0 | 4.02e-01 | 20 | 0 | 0 | 4.02e-01 | 20 | 0 |
| ccd_screw_interval_ee | EE screw CCD (interval) | degenerate | 1280 | 0 | 1.32e-14 | 0 | 0 | 0 | 1.32e-14 | 0 | 0 |
| ccd_screw_interval_ee | EE screw CCD (interval) | random | 4000 | 0 | 2.93e-01 | 7 | 0 | 0 | 2.93e-01 | 7 | 0 |
| ccd_screw_interval_pt | PT screw CCD (interval) | degenerate | 2048 | 0 | 1.57e-14 | 0 | 0 | 0 | 1.57e-14 | 0 | 0 |
| ccd_screw_interval_pt | PT screw CCD (interval) | random | 4000 | 0 | 3.56e-01 | 1 | 0 | 0 | 3.56e-01 | 1 | 0 |
| ccd_screw_pe | PE screw CCD (whole step) | degenerate | 256 | 0 | 2.55e-15 | 0 | 0 | 0 | 2.55e-15 | 0 | 0 |
| ccd_screw_pe | PE screw CCD (whole step) | random | 4000 | 0 | 6.05e-14 | 0 | 0 | 0 | 6.05e-14 | 0 | 0 |
| ccd_screw_pp | PP screw CCD (whole step) | degenerate | 64 | 0 | 2.22e-16 | 0 | 0 | 0 | 2.22e-16 | 0 | 0 |
| ccd_screw_pp | PP screw CCD (whole step) | random | 4000 | 0 | 9.99e-16 | 0 | 0 | 0 | 9.99e-16 | 0 | 0 |
| ccd_screw_pt | PT screw CCD (whole step) | degenerate | 2048 | 0 | 2.51e-01 | 0 | 0 | 0 | 2.51e-01 | 0 | 0 |
| ccd_screw_pt | PT screw CCD (whole step) | random | 4000 | 0 | 3.56e-01 | 3 | 0 | 0 | 3.56e-01 | 3 | 0 |
| closest_ee | EE closest points | degenerate | 3632 | 0 | 1.55e-01 | 179 | 0 | 0 | 1.55e-01 | 179 | 0 |
| closest_ee | EE closest points | random | 4000 | 0 | 6.67e-14 | 0 | 0 | 0 | 6.67e-14 | 0 | 0 |
| closest_ee | EE closest points | real_cloth_stack | 30000 | 0 | 4.17e-15 | 0 | 0 | 0 | 4.17e-15 | 0 | 0 |
| closest_ee | EE closest points | real_kimono | 30000 | 0 | 1.02e-15 | 0 | 0 | 0 | 1.02e-15 | 0 | 0 |
| closest_pe | PE closest point | degenerate | 336 | 0 | 0.00e+00 | 0 | 0 | 0 | 0.00e+00 | 0 | 0 |
| closest_pe | PE closest point | random | 4000 | 0 | 5.55e-16 | 0 | 0 | 0 | 5.55e-16 | 0 | 0 |
| closest_pt | PT closest point | degenerate | 2580 | 0 | 0.00e+00 | 0 | 0 | 0 | 0.00e+00 | 0 | 0 |
| closest_pt | PT closest point | random | 4000 | 0 | 1.11e-15 | 0 | 0 | 0 | 1.11e-15 | 0 | 0 |
| closest_pt | PT closest point | real_cloth_stack | 30000 | 0 | 4.28e-15 | 0 | 0 | 0 | 4.28e-15 | 0 | 0 |
| closest_pt | PT closest point | real_kimono | 30000 | 0 | 3.87e-16 | 0 | 0 | 0 | 3.87e-16 | 0 | 0 |
| ee_distance | EE distance flag, flagged distance2 and gradient | degenerate | 3632 | 0 | 3.13e-13 | 0 | 0 | 0 | 3.13e-13 | 0 | 0 |
| ee_distance | EE distance flag, flagged distance2 and gradient | random | 4000 | 0 | 2.64e-15 | 0 | 0 | 0 | 2.64e-15 | 0 | 0 |
| ee_distance | EE distance flag, flagged distance2 and gradient | real_cloth_stack | 30000 | 0 | 1.90e-15 | 0 | 0 | 0 | 1.90e-15 | 0 | 0 |
| ee_distance | EE distance flag, flagged distance2 and gradient | real_kimono | 30000 | 0 | 2.83e-16 | 0 | 0 | 0 | 2.83e-16 | 0 | 0 |
| ee_distance_type | EE distance type (parallel projection) | degenerate | 3632 | 0 | - | 140 | 0 | 0 | - | 140 | 0 |
| ee_distance_type | EE distance type (parallel projection) | random | 4000 | 0 | - | 0 | 0 | 0 | - | 0 | 0 |
| ee_distance_type | EE distance type (parallel projection) | real_cloth_stack | 30000 | 0 | - | 0 | 0 | 0 | - | 0 | 0 |
| ee_distance_type | EE distance type (parallel projection) | real_kimono | 30000 | 0 | - | 0 | 0 | 0 | - | 0 | 0 |
| frame_ee | EE tangent basis and closest-point weights | degenerate | 868 | 0 | 2.09e-13 | 0 | 0 | 0 | 2.09e-13 | 0 | 0 |
| frame_ee | EE tangent basis and closest-point weights | random | 3998 | 0 | 2.48e-13 | 0 | 0 | 0 | 2.48e-13 | 0 | 0 |
| frame_ee | EE tangent basis and closest-point weights | real_cloth_stack | 19905 | 0 | 5.46e-15 | 0 | 0 | 0 | 5.46e-15 | 0 | 0 |
| frame_ee | EE tangent basis and closest-point weights | real_kimono | 19902 | 0 | 4.60e-13 | 0 | 0 | 0 | 4.60e-13 | 0 | 0 |
| frame_pe | PE tangent basis and closest-point weights | degenerate | 256 | 0 | 0.00e+00 | 0 | 0 | 0 | 0.00e+00 | 0 | 0 |
| frame_pe | PE tangent basis and closest-point weights | random | 4000 | 0 | 1.11e-15 | 0 | 0 | 0 | 1.11e-15 | 0 | 0 |
| frame_pe | PE tangent basis and closest-point weights | real_cloth_stack | 17883 | 0 | 2.55e-14 | 0 | 0 | 0 | 2.55e-14 | 0 | 0 |
| frame_pe | PE tangent basis and closest-point weights | real_kimono | 12741 | 0 | 1.65e-14 | 0 | 0 | 0 | 1.65e-14 | 0 | 0 |
| frame_ph | PH tangent basis (projector) | degenerate | 1476 | 0 | 0.00e+00 | 0 | 0 | 0 | 0.00e+00 | 0 | 0 |
| frame_ph | PH tangent basis (projector) | random | 4000 | 0 | 8.88e-16 | 0 | 0 | 0 | 8.88e-16 | 0 | 0 |
| frame_pp | PP tangent basis (projector) | degenerate | 64 | 0 | 4.44e-16 | 0 | 0 | 0 | 4.44e-16 | 0 | 0 |
| frame_pp | PP tangent basis (projector) | random | 4000 | 0 | 9.99e-16 | 0 | 0 | 0 | 9.99e-16 | 0 | 0 |
| frame_pp | PP tangent basis (projector) | real_cloth_stack | 1404 | 0 | 8.88e-16 | 0 | 0 | 0 | 8.88e-16 | 0 | 0 |
| frame_pp | PP tangent basis (projector) | real_kimono | 882 | 0 | 8.88e-16 | 0 | 0 | 0 | 8.88e-16 | 0 | 0 |
| frame_pt | PT tangent basis and closest-point weights | degenerate | 2048 | 0 | 0.00e+00 | 0 | 0 | 0 | 0.00e+00 | 0 | 0 |
| frame_pt | PT tangent basis and closest-point weights | random | 4000 | 0 | 1.61e-13 | 0 | 0 | 0 | 1.61e-13 | 0 | 0 |
| frame_pt | PT tangent basis and closest-point weights | real_cloth_stack | 20807 | 0 | 1.36e-15 | 0 | 0 | 0 | 1.36e-15 | 0 | 0 |
| frame_pt | PT tangent basis and closest-point weights | real_kimono | 26473 | 0 | 1.78e-15 | 0 | 0 | 0 | 1.78e-15 | 0 | 0 |
| friction_ee | EE friction energy/gradient/Hessian | degenerate | 868 | 0 | 6.00e-14 | 0 | 0 | 0 | 6.00e-14 | 0 | 0 |
| friction_ee | EE friction energy/gradient/Hessian | random | 3998 | 0 | 9.67e-14 | 0 | 0 | 0 | 9.67e-14 | 0 | 0 |
| friction_ee | EE friction energy/gradient/Hessian | real_cloth_stack | 19905 | 0 | 9.27e-15 | 0 | 0 | 0 | 9.27e-15 | 0 | 0 |
| friction_ee | EE friction energy/gradient/Hessian | real_kimono | 19902 | 0 | 1.66e-14 | 0 | 0 | 0 | 1.66e-14 | 0 | 0 |
| friction_pe | PE friction energy/gradient/Hessian | degenerate | 256 | 0 | 1.99e-16 | 0 | 0 | 0 | 1.99e-16 | 0 | 0 |
| friction_pe | PE friction energy/gradient/Hessian | random | 4000 | 0 | 2.57e-15 | 0 | 0 | 0 | 2.57e-15 | 0 | 0 |
| friction_pe | PE friction energy/gradient/Hessian | real_cloth_stack | 17883 | 0 | 1.82e-14 | 0 | 0 | 0 | 1.82e-14 | 0 | 0 |
| friction_pe | PE friction energy/gradient/Hessian | real_kimono | 12741 | 0 | 4.37e-15 | 0 | 0 | 0 | 4.37e-15 | 0 | 0 |
| friction_ph | PH friction energy/gradient/Hessian | degenerate | 1476 | 0 | 7.43e-17 | 0 | 0 | 0 | 7.43e-17 | 0 | 0 |
| friction_ph | PH friction energy/gradient/Hessian | random | 4000 | 0 | 2.13e-13 | 0 | 0 | 0 | 2.13e-13 | 0 | 0 |
| friction_pp | PP friction energy/gradient/Hessian | degenerate | 64 | 0 | 5.06e-16 | 0 | 0 | 0 | 5.06e-16 | 0 | 0 |
| friction_pp | PP friction energy/gradient/Hessian | random | 4000 | 0 | 1.87e-15 | 0 | 0 | 0 | 1.87e-15 | 0 | 0 |
| friction_pp | PP friction energy/gradient/Hessian | real_cloth_stack | 1404 | 0 | 1.04e-14 | 0 | 0 | 0 | 1.04e-14 | 0 | 0 |
| friction_pp | PP friction energy/gradient/Hessian | real_kimono | 882 | 0 | 3.44e-16 | 0 | 0 | 0 | 3.44e-16 | 0 | 0 |
| friction_pt | PT friction energy/gradient/Hessian | degenerate | 2048 | 0 | 2.22e-15 | 0 | 0 | 0 | 2.22e-15 | 0 | 0 |
| friction_pt | PT friction energy/gradient/Hessian | random | 4000 | 0 | 3.38e-14 | 0 | 0 | 0 | 3.38e-14 | 0 | 0 |
| friction_pt | PT friction energy/gradient/Hessian | real_cloth_stack | 20807 | 0 | 4.97e-15 | 0 | 0 | 0 | 4.97e-15 | 0 | 0 |
| friction_pt | PT friction energy/gradient/Hessian | real_kimono | 26473 | 0 | 5.33e-15 | 0 | 0 | 0 | 5.33e-15 | 0 | 0 |
| intersect | edge-triangle intersection | degenerate | 800 | 0 | - | 0 | 0 | 0 | - | 0 | 0 |
| intersect | edge-triangle intersection | random | 4000 | 0 | - | 0 | 0 | 0 | - | 0 | 0 |
| mollified_barrier | mollified barrier | degenerate | 4000 | 0 | 7.83e-07 | 0 | 0 | 0 | 7.83e-07 | 0 | 0 |
| mollified_barrier | mollified barrier | random | 4000 | 0 | 1.81e-16 | 0 | 0 | 0 | 1.81e-16 | 0 | 0 |
| mollifier | mollifier threshold and edge cross norm2 | degenerate | 3632 | 0 | 2.27e-16 | 0 | 0 | 0 | 2.27e-16 | 0 | 0 |
| mollifier | mollifier threshold and edge cross norm2 | random | 4000 | 0 | 2.38e-16 | 0 | 0 | 0 | 2.38e-16 | 0 | 0 |
| mollifier | mollifier threshold and edge cross norm2 | real_cloth_stack | 30000 | 0 | 2.48e-20 | 0 | 0 | 0 | 2.48e-20 | 0 | 0 |
| mollifier | mollifier threshold and edge cross norm2 | real_kimono | 30000 | 0 | 4.84e-16 | 0 | 0 | 0 | 4.84e-16 | 0 | 0 |
| pair_ee | EE pair d_hat, thickness, area weight | random | 4000 | 0 | 0.00e+00 | 0 | 0 | 0 | 0.00e+00 | 0 | 0 |
| pair_pe | PE pair d_hat, thickness, area weight | random | 4000 | 0 | 0.00e+00 | 0 | 0 | 0 | 0.00e+00 | 0 | 0 |
| pair_pp | PP pair d_hat, thickness, area weight | random | 4000 | 0 | 0.00e+00 | 0 | 0 | 0 | 0.00e+00 | 0 | 0 |
| pair_pt | PT pair d_hat, thickness, area weight | random | 4000 | 0 | 0.00e+00 | 0 | 0 | 0 | 0.00e+00 | 0 | 0 |
| pe_distance | PE distance flag, flagged distance2 and gradient | degenerate | 336 | 0 | 0.00e+00 | 0 | 0 | 0 | 0.00e+00 | 0 | 0 |
| pe_distance | PE distance flag, flagged distance2 and gradient | random | 4000 | 0 | 4.81e-15 | 0 | 0 | 0 | 4.81e-15 | 0 | 0 |
| pe_distance | PE distance flag, flagged distance2 and gradient | real_cloth_stack | 17883 | 0 | 2.08e-15 | 0 | 0 | 0 | 2.08e-15 | 0 | 0 |
| pe_distance | PE distance flag, flagged distance2 and gradient | real_kimono | 12741 | 0 | 7.20e-15 | 0 | 0 | 0 | 7.20e-15 | 0 | 0 |
| pt_distance | PT distance flag, flagged distance2 and gradient | degenerate | 2580 | 0 | 0.00e+00 | 0 | 0 | 0 | 0.00e+00 | 0 | 0 |
| pt_distance | PT distance flag, flagged distance2 and gradient | random | 4000 | 0 | 1.95e-15 | 0 | 0 | 0 | 1.95e-15 | 0 | 0 |
| pt_distance | PT distance flag, flagged distance2 and gradient | real_cloth_stack | 30000 | 0 | 1.92e-15 | 0 | 0 | 0 | 1.92e-15 | 0 | 0 |
| pt_distance | PT distance flag, flagged distance2 and gradient | real_kimono | 30000 | 0 | 2.15e-16 | 0 | 0 | 0 | 2.15e-16 | 0 | 0 |

Totals: cuda_fast-math-false: 0 failing rows of 1452423, cpu_fast-math-false: 0 failing rows of 1452423
