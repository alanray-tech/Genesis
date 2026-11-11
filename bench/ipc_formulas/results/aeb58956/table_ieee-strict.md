# IPC formula parity: ieee-strict

Per case and input set: rows, then per backend the failing rows, the largest real error over the scale of its output (max |g - c| / s), and the rows that passed by the order clause and by the referee (compare.py).

| case | function | set | rows | cuda_fast-math-false failures | cuda_fast-math-false max err / s | cuda_fast-math-false order rows | cuda_fast-math-false referee rows | cpu_fast-math-false failures | cpu_fast-math-false max err / s | cpu_fast-math-false order rows | cpu_fast-math-false referee rows |
|---|---|---|---|---|---|---|---|---|---|---|---|
| barrier | barrier, derivatives, normal force | degenerate | 4000 | 0 | 0.00e+00 | 0 | 0 | 0 | 9.01e-17 | 0 | 0 |
| barrier | barrier, derivatives, normal force | random | 4000 | 0 | 0.00e+00 | 0 | 0 | 0 | 1.26e-20 | 0 | 0 |
| barrier_ee | EE barrier gradient/Hessian (mollifier and flag dispatch) | degenerate | 5280 | 0 | 2.59e-16 | 0 | 0 | 0 | 2.59e-16 | 0 | 0 |
| barrier_ee | EE barrier gradient/Hessian (mollifier and flag dispatch) | mollifier_sweep | 4000 | 0 | 9.67e-15 | 0 | 0 | 0 | 9.67e-15 | 0 | 0 |
| barrier_ee | EE barrier gradient/Hessian (mollifier and flag dispatch) | parallel | 576 | 0 | 3.46e-16 | 0 | 0 | 0 | 3.46e-16 | 0 | 0 |
| barrier_ee | EE barrier gradient/Hessian (mollifier and flag dispatch) | random | 4000 | 0 | 1.15e-13 | 0 | 0 | 0 | 1.15e-13 | 0 | 0 |
| barrier_ee | EE barrier gradient/Hessian (mollifier and flag dispatch) | real_cloth_stack | 30000 | 0 | 6.11e-15 | 0 | 0 | 0 | 6.11e-15 | 0 | 0 |
| barrier_ee | EE barrier gradient/Hessian (mollifier and flag dispatch) | real_cloth_stack_mollified | 18457 | 0 | 2.03e-14 | 0 | 0 | 0 | 2.03e-14 | 0 | 0 |
| barrier_ee | EE barrier gradient/Hessian (mollifier and flag dispatch) | real_kimono | 30000 | 0 | 2.02e-16 | 0 | 0 | 0 | 2.02e-16 | 0 | 0 |
| barrier_ee | EE barrier gradient/Hessian (mollifier and flag dispatch) | real_kimono_mollified | 514 | 0 | 1.41e-15 | 0 | 0 | 0 | 1.41e-15 | 0 | 0 |
| barrier_pe | PE barrier gradient/Hessian (flag dispatch) | degenerate | 256 | 0 | 2.25e-16 | 0 | 0 | 0 | 2.25e-16 | 0 | 0 |
| barrier_pe | PE barrier gradient/Hessian (flag dispatch) | random | 4000 | 0 | 5.48e-15 | 0 | 0 | 0 | 5.48e-15 | 0 | 0 |
| barrier_pe | PE barrier gradient/Hessian (flag dispatch) | real_cloth_stack | 17883 | 0 | 1.84e-14 | 0 | 0 | 0 | 1.84e-14 | 0 | 0 |
| barrier_pe | PE barrier gradient/Hessian (flag dispatch) | real_kimono | 12741 | 0 | 5.36e-16 | 0 | 0 | 0 | 5.36e-16 | 0 | 0 |
| barrier_ph | PH barrier gradient/Hessian | degenerate | 1476 | 0 | 0.00e+00 | 0 | 0 | 0 | 0.00e+00 | 0 | 0 |
| barrier_ph | PH barrier gradient/Hessian | random | 4000 | 0 | 9.00e-15 | 0 | 0 | 0 | 9.00e-15 | 0 | 0 |
| barrier_pp | PP barrier gradient/Hessian | degenerate | 64 | 0 | 1.42e-16 | 0 | 0 | 0 | 1.42e-16 | 0 | 0 |
| barrier_pp | PP barrier gradient/Hessian | random | 4000 | 0 | 1.71e-16 | 0 | 0 | 0 | 1.71e-16 | 0 | 0 |
| barrier_pp | PP barrier gradient/Hessian | real_cloth_stack | 1404 | 0 | 5.78e-15 | 0 | 0 | 0 | 5.78e-15 | 0 | 0 |
| barrier_pp | PP barrier gradient/Hessian | real_kimono | 882 | 0 | 2.02e-16 | 0 | 0 | 0 | 2.02e-16 | 0 | 0 |
| barrier_pt | PT barrier gradient/Hessian (flag dispatch) | degenerate | 2048 | 0 | 4.47e-16 | 0 | 0 | 0 | 4.47e-16 | 0 | 0 |
| barrier_pt | PT barrier gradient/Hessian (flag dispatch) | random | 4000 | 0 | 5.53e-14 | 0 | 0 | 0 | 5.53e-14 | 0 | 0 |
| barrier_pt | PT barrier gradient/Hessian (flag dispatch) | real_cloth_stack | 30000 | 0 | 5.09e-15 | 0 | 0 | 0 | 5.09e-15 | 0 | 0 |
| barrier_pt | PT barrier gradient/Hessian (flag dispatch) | real_kimono | 30000 | 0 | 2.00e-16 | 0 | 0 | 0 | 2.00e-16 | 0 | 0 |
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
| ccd_additive_ee | EE additive CCD | degenerate | 4560 | 34 | 1.61e-10 | 0 | 0 | 34 | 1.61e-10 | 0 | 0 |
| ccd_additive_ee | EE additive CCD | random | 4000 | 0 | 1.01e-13 | 0 | 0 | 0 | 1.01e-13 | 0 | 0 |
| ccd_additive_ee | EE additive CCD | real_cloth_stack | 30000 | 0 | 0.00e+00 | 0 | 0 | 0 | 0.00e+00 | 0 | 0 |
| ccd_additive_ee | EE additive CCD | real_kimono | 30000 | 0 | 1.33e-15 | 0 | 0 | 0 | 1.33e-15 | 0 | 0 |
| ccd_additive_pe | PE additive CCD | degenerate | 512 | 0 | 2.61e-13 | 0 | 0 | 0 | 2.61e-13 | 0 | 0 |
| ccd_additive_pe | PE additive CCD | random | 4000 | 0 | 3.89e-16 | 0 | 0 | 0 | 3.89e-16 | 0 | 0 |
| ccd_additive_pp | PP additive CCD | degenerate | 128 | 0 | 2.46e-14 | 0 | 0 | 0 | 2.46e-14 | 0 | 0 |
| ccd_additive_pp | PP additive CCD | random | 4000 | 0 | 2.78e-16 | 0 | 0 | 0 | 2.78e-16 | 0 | 0 |
| ccd_additive_pt | PT additive CCD | degenerate | 4096 | 1 | 4.50e-12 | 0 | 0 | 1 | 4.50e-12 | 0 | 0 |
| ccd_additive_pt | PT additive CCD | random | 4000 | 0 | 6.05e-14 | 0 | 0 | 0 | 6.05e-14 | 0 | 0 |
| ccd_additive_pt | PT additive CCD | real_cloth_stack | 30000 | 0 | 0.00e+00 | 0 | 0 | 0 | 0.00e+00 | 0 | 0 |
| ccd_additive_pt | PT additive CCD | real_kimono | 30000 | 0 | 3.33e-15 | 0 | 0 | 0 | 3.33e-15 | 0 | 0 |
| ccd_directional_ee | EE directional CCD | degenerate | 4560 | 250 | 1.15e-07 | 0 | 0 | 250 | 1.15e-07 | 0 | 0 |
| ccd_directional_ee | EE directional CCD | random | 4000 | 9 | 2.14e-11 | 0 | 0 | 9 | 2.14e-11 | 0 | 0 |
| ccd_directional_ee | EE directional CCD | real_cloth_stack | 30000 | 0 | 0.00e+00 | 0 | 0 | 0 | 0.00e+00 | 0 | 0 |
| ccd_directional_ee | EE directional CCD | real_kimono | 30000 | 3 | 5.50e-12 | 0 | 0 | 3 | 5.50e-12 | 0 | 0 |
| ccd_directional_pe | PE directional CCD | degenerate | 512 | 0 | 7.62e-13 | 0 | 0 | 0 | 7.62e-13 | 0 | 0 |
| ccd_directional_pe | PE directional CCD | random | 4000 | 0 | 2.26e-14 | 0 | 0 | 0 | 2.26e-14 | 0 | 0 |
| ccd_directional_pp | PP directional CCD | degenerate | 128 | 0 | 6.97e-14 | 0 | 0 | 0 | 6.97e-14 | 0 | 0 |
| ccd_directional_pp | PP directional CCD | random | 4000 | 0 | 4.44e-16 | 0 | 0 | 0 | 4.44e-16 | 0 | 0 |
| ccd_directional_pt | PT directional CCD | degenerate | 4096 | 4 | 2.25e-12 | 0 | 0 | 4 | 2.25e-12 | 0 | 0 |
| ccd_directional_pt | PT directional CCD | random | 4000 | 1 | 2.37e-11 | 0 | 0 | 1 | 2.37e-11 | 0 | 0 |
| ccd_directional_pt | PT directional CCD | real_cloth_stack | 30000 | 0 | 0.00e+00 | 0 | 0 | 0 | 0.00e+00 | 0 | 0 |
| ccd_directional_pt | PT directional CCD | real_kimono | 30000 | 0 | 7.92e-14 | 0 | 0 | 0 | 7.92e-14 | 0 | 0 |
| ccd_ph | PH CCD (linear) | degenerate | 1476 | 0 | 0.00e+00 | 0 | 0 | 0 | 0.00e+00 | 0 | 0 |
| ccd_ph | PH CCD (linear) | random | 4000 | 0 | 2.79e-14 | 0 | 0 | 0 | 2.79e-14 | 0 | 0 |
| ccd_ph_screw | PH screw CCD | degenerate | 1476 | 0 | 4.44e-16 | 0 | 0 | 0 | 4.44e-16 | 0 | 0 |
| ccd_ph_screw | PH screw CCD | random | 4000 | 0 | 1.71e-14 | 0 | 0 | 0 | 1.71e-14 | 0 | 0 |
| ccd_screw_ee | EE screw CCD (whole step) | degenerate | 1280 | 0 | 1.76e-14 | 0 | 0 | 0 | 1.76e-14 | 0 | 0 |
| ccd_screw_ee | EE screw CCD (whole step) | random | 4000 | 32 | 3.02e-01 | 0 | 0 | 32 | 3.02e-01 | 0 | 0 |
| ccd_screw_interval_ee | EE screw CCD (interval) | degenerate | 1280 | 0 | 1.76e-14 | 0 | 0 | 0 | 1.76e-14 | 0 | 0 |
| ccd_screw_interval_ee | EE screw CCD (interval) | random | 4000 | 9 | 2.08e-01 | 0 | 0 | 9 | 2.08e-01 | 0 | 0 |
| ccd_screw_interval_pt | PT screw CCD (interval) | degenerate | 2048 | 0 | 5.61e-15 | 0 | 0 | 0 | 5.61e-15 | 0 | 0 |
| ccd_screw_interval_pt | PT screw CCD (interval) | random | 4000 | 5 | 3.56e-01 | 0 | 0 | 5 | 3.56e-01 | 0 | 0 |
| ccd_screw_pe | PE screw CCD (whole step) | degenerate | 256 | 0 | 1.33e-15 | 0 | 0 | 0 | 1.33e-15 | 0 | 0 |
| ccd_screw_pe | PE screw CCD (whole step) | random | 4000 | 0 | 1.50e-14 | 0 | 0 | 0 | 1.50e-14 | 0 | 0 |
| ccd_screw_pp | PP screw CCD (whole step) | degenerate | 64 | 0 | 2.22e-16 | 0 | 0 | 0 | 2.22e-16 | 0 | 0 |
| ccd_screw_pp | PP screw CCD (whole step) | random | 4000 | 0 | 4.44e-16 | 0 | 0 | 0 | 4.44e-16 | 0 | 0 |
| ccd_screw_pt | PT screw CCD (whole step) | degenerate | 2048 | 0 | 1.10e-14 | 0 | 0 | 0 | 1.10e-14 | 0 | 0 |
| ccd_screw_pt | PT screw CCD (whole step) | random | 4000 | 8 | 3.56e-01 | 0 | 0 | 8 | 3.56e-01 | 0 | 0 |
| closest_ee | EE closest points | degenerate | 3632 | 403 | 8.51e-02 | 0 | 0 | 403 | 8.51e-02 | 0 | 0 |
| closest_ee | EE closest points | random | 4000 | 0 | 7.58e-14 | 0 | 0 | 0 | 7.58e-14 | 0 | 0 |
| closest_ee | EE closest points | real_cloth_stack | 30000 | 0 | 4.17e-15 | 0 | 0 | 0 | 4.17e-15 | 0 | 0 |
| closest_ee | EE closest points | real_kimono | 30000 | 0 | 6.59e-16 | 0 | 0 | 0 | 6.59e-16 | 0 | 0 |
| closest_pe | PE closest point | degenerate | 336 | 0 | 0.00e+00 | 0 | 0 | 0 | 0.00e+00 | 0 | 0 |
| closest_pe | PE closest point | random | 4000 | 0 | 4.44e-16 | 0 | 0 | 0 | 4.44e-16 | 0 | 0 |
| closest_pt | PT closest point | degenerate | 2580 | 0 | 0.00e+00 | 0 | 0 | 0 | 0.00e+00 | 0 | 0 |
| closest_pt | PT closest point | random | 4000 | 0 | 1.22e-15 | 0 | 0 | 0 | 1.22e-15 | 0 | 0 |
| closest_pt | PT closest point | real_cloth_stack | 30000 | 0 | 4.28e-15 | 0 | 0 | 0 | 4.28e-15 | 0 | 0 |
| closest_pt | PT closest point | real_kimono | 30000 | 0 | 3.87e-16 | 0 | 0 | 0 | 3.87e-16 | 0 | 0 |
| ee_distance | EE distance flag, flagged distance2 and gradient | degenerate | 3632 | 0 | 3.17e-15 | 0 | 0 | 0 | 3.17e-15 | 0 | 0 |
| ee_distance | EE distance flag, flagged distance2 and gradient | random | 4000 | 0 | 1.64e-15 | 0 | 0 | 0 | 1.64e-15 | 0 | 0 |
| ee_distance | EE distance flag, flagged distance2 and gradient | real_cloth_stack | 30000 | 0 | 1.32e-15 | 0 | 0 | 0 | 1.32e-15 | 0 | 0 |
| ee_distance | EE distance flag, flagged distance2 and gradient | real_kimono | 30000 | 0 | 2.17e-16 | 0 | 0 | 0 | 2.17e-16 | 0 | 0 |
| ee_distance_type | EE distance type (parallel projection) | degenerate | 3632 | 230 | - | 0 | 0 | 230 | - | 0 | 0 |
| ee_distance_type | EE distance type (parallel projection) | random | 4000 | 0 | - | 0 | 0 | 0 | - | 0 | 0 |
| ee_distance_type | EE distance type (parallel projection) | real_cloth_stack | 30000 | 0 | - | 0 | 0 | 0 | - | 0 | 0 |
| ee_distance_type | EE distance type (parallel projection) | real_kimono | 30000 | 0 | - | 0 | 0 | 0 | - | 0 | 0 |
| frame_ee | EE tangent basis and closest-point weights | degenerate | 868 | 0 | 2.27e-13 | 0 | 0 | 0 | 2.27e-13 | 0 | 0 |
| frame_ee | EE tangent basis and closest-point weights | random | 3998 | 0 | 2.16e-13 | 0 | 0 | 0 | 2.16e-13 | 0 | 0 |
| frame_ee | EE tangent basis and closest-point weights | real_cloth_stack | 19905 | 0 | 5.33e-15 | 0 | 0 | 0 | 5.33e-15 | 0 | 0 |
| frame_ee | EE tangent basis and closest-point weights | real_kimono | 19902 | 0 | 4.55e-13 | 0 | 0 | 0 | 4.55e-13 | 0 | 0 |
| frame_pe | PE tangent basis and closest-point weights | degenerate | 256 | 0 | 0.00e+00 | 0 | 0 | 0 | 0.00e+00 | 0 | 0 |
| frame_pe | PE tangent basis and closest-point weights | random | 4000 | 0 | 3.33e-16 | 0 | 0 | 0 | 3.33e-16 | 0 | 0 |
| frame_pe | PE tangent basis and closest-point weights | real_cloth_stack | 17883 | 0 | 3.33e-16 | 0 | 0 | 0 | 3.33e-16 | 0 | 0 |
| frame_pe | PE tangent basis and closest-point weights | real_kimono | 12741 | 0 | 4.44e-16 | 0 | 0 | 0 | 4.44e-16 | 0 | 0 |
| frame_ph | PH tangent basis (projector) | degenerate | 1476 | 0 | 0.00e+00 | 0 | 0 | 0 | 0.00e+00 | 0 | 0 |
| frame_ph | PH tangent basis (projector) | random | 4000 | 0 | 8.88e-16 | 0 | 0 | 0 | 8.88e-16 | 0 | 0 |
| frame_pp | PP tangent basis (projector) | degenerate | 64 | 0 | 4.44e-16 | 0 | 0 | 0 | 4.44e-16 | 0 | 0 |
| frame_pp | PP tangent basis (projector) | random | 4000 | 0 | 9.99e-16 | 0 | 0 | 0 | 9.99e-16 | 0 | 0 |
| frame_pp | PP tangent basis (projector) | real_cloth_stack | 1404 | 0 | 9.99e-16 | 0 | 0 | 0 | 9.99e-16 | 0 | 0 |
| frame_pp | PP tangent basis (projector) | real_kimono | 882 | 0 | 8.88e-16 | 0 | 0 | 0 | 8.88e-16 | 0 | 0 |
| frame_pt | PT tangent basis and closest-point weights | degenerate | 2048 | 0 | 0.00e+00 | 0 | 0 | 0 | 0.00e+00 | 0 | 0 |
| frame_pt | PT tangent basis and closest-point weights | random | 4000 | 0 | 1.07e-13 | 0 | 0 | 0 | 1.07e-13 | 0 | 0 |
| frame_pt | PT tangent basis and closest-point weights | real_cloth_stack | 20807 | 0 | 1.22e-15 | 0 | 0 | 0 | 1.22e-15 | 0 | 0 |
| frame_pt | PT tangent basis and closest-point weights | real_kimono | 26473 | 0 | 2.00e-15 | 0 | 0 | 0 | 2.00e-15 | 0 | 0 |
| friction_ee | EE friction energy/gradient/Hessian | degenerate | 868 | 0 | 4.91e-14 | 0 | 0 | 0 | 4.91e-14 | 0 | 0 |
| friction_ee | EE friction energy/gradient/Hessian | random | 3998 | 0 | 4.80e-14 | 0 | 0 | 0 | 4.80e-14 | 0 | 0 |
| friction_ee | EE friction energy/gradient/Hessian | real_cloth_stack | 19905 | 0 | 5.25e-15 | 0 | 0 | 0 | 5.25e-15 | 0 | 0 |
| friction_ee | EE friction energy/gradient/Hessian | real_kimono | 19902 | 0 | 1.33e-14 | 0 | 0 | 0 | 1.33e-14 | 0 | 0 |
| friction_pe | PE friction energy/gradient/Hessian | degenerate | 256 | 0 | 1.99e-16 | 0 | 0 | 0 | 1.99e-16 | 0 | 0 |
| friction_pe | PE friction energy/gradient/Hessian | random | 4000 | 0 | 2.33e-15 | 0 | 0 | 0 | 2.33e-15 | 0 | 0 |
| friction_pe | PE friction energy/gradient/Hessian | real_cloth_stack | 17883 | 0 | 1.82e-14 | 0 | 0 | 0 | 1.82e-14 | 0 | 0 |
| friction_pe | PE friction energy/gradient/Hessian | real_kimono | 12741 | 0 | 3.98e-15 | 0 | 0 | 0 | 3.98e-15 | 0 | 0 |
| friction_ph | PH friction energy/gradient/Hessian | degenerate | 1476 | 0 | 0.00e+00 | 0 | 0 | 0 | 4.07e-24 | 0 | 0 |
| friction_ph | PH friction energy/gradient/Hessian | random | 4000 | 0 | 7.47e-14 | 0 | 0 | 0 | 7.47e-14 | 0 | 0 |
| friction_pp | PP friction energy/gradient/Hessian | degenerate | 64 | 0 | 3.79e-16 | 0 | 0 | 0 | 3.79e-16 | 0 | 0 |
| friction_pp | PP friction energy/gradient/Hessian | random | 4000 | 0 | 2.10e-16 | 0 | 0 | 0 | 2.10e-16 | 0 | 0 |
| friction_pp | PP friction energy/gradient/Hessian | real_cloth_stack | 1404 | 0 | 7.08e-15 | 0 | 0 | 0 | 7.08e-15 | 0 | 0 |
| friction_pp | PP friction energy/gradient/Hessian | real_kimono | 882 | 0 | 4.59e-16 | 0 | 0 | 0 | 4.59e-16 | 0 | 0 |
| friction_pt | PT friction energy/gradient/Hessian | degenerate | 2048 | 0 | 2.22e-15 | 0 | 0 | 0 | 2.22e-15 | 0 | 0 |
| friction_pt | PT friction energy/gradient/Hessian | random | 4000 | 0 | 2.40e-14 | 0 | 0 | 0 | 2.40e-14 | 0 | 0 |
| friction_pt | PT friction energy/gradient/Hessian | real_cloth_stack | 20807 | 0 | 6.35e-15 | 0 | 0 | 0 | 6.35e-15 | 0 | 0 |
| friction_pt | PT friction energy/gradient/Hessian | real_kimono | 26473 | 0 | 1.35e-14 | 0 | 0 | 0 | 1.35e-14 | 0 | 0 |
| intersect | edge-triangle intersection | degenerate | 800 | 0 | - | 0 | 0 | 0 | - | 0 | 0 |
| intersect | edge-triangle intersection | random | 4000 | 0 | - | 0 | 0 | 0 | - | 0 | 0 |
| mollified_barrier | mollified barrier | degenerate | 4000 | 0 | 0.00e+00 | 0 | 0 | 0 | 3.85e-19 | 0 | 0 |
| mollified_barrier | mollified barrier | random | 4000 | 0 | 0.00e+00 | 0 | 0 | 0 | 1.48e-20 | 0 | 0 |
| mollifier | mollifier threshold and edge cross norm2 | degenerate | 3632 | 0 | 2.27e-16 | 0 | 0 | 0 | 2.27e-16 | 0 | 0 |
| mollifier | mollifier threshold and edge cross norm2 | random | 4000 | 0 | 3.10e-16 | 0 | 0 | 0 | 3.10e-16 | 0 | 0 |
| mollifier | mollifier threshold and edge cross norm2 | real_cloth_stack | 30000 | 0 | 8.27e-21 | 0 | 0 | 0 | 8.27e-21 | 0 | 0 |
| mollifier | mollifier threshold and edge cross norm2 | real_kimono | 30000 | 0 | 2.46e-16 | 0 | 0 | 0 | 2.46e-16 | 0 | 0 |
| pair_ee | EE pair d_hat, thickness, area weight | random | 4000 | 0 | 0.00e+00 | 0 | 0 | 0 | 0.00e+00 | 0 | 0 |
| pair_pe | PE pair d_hat, thickness, area weight | random | 4000 | 0 | 0.00e+00 | 0 | 0 | 0 | 0.00e+00 | 0 | 0 |
| pair_pp | PP pair d_hat, thickness, area weight | random | 4000 | 0 | 0.00e+00 | 0 | 0 | 0 | 0.00e+00 | 0 | 0 |
| pair_pt | PT pair d_hat, thickness, area weight | random | 4000 | 0 | 0.00e+00 | 0 | 0 | 0 | 0.00e+00 | 0 | 0 |
| pe_distance | PE distance flag, flagged distance2 and gradient | degenerate | 336 | 0 | 0.00e+00 | 0 | 0 | 0 | 0.00e+00 | 0 | 0 |
| pe_distance | PE distance flag, flagged distance2 and gradient | random | 4000 | 0 | 1.64e-15 | 0 | 0 | 0 | 1.64e-15 | 0 | 0 |
| pe_distance | PE distance flag, flagged distance2 and gradient | real_cloth_stack | 17883 | 0 | 5.78e-16 | 0 | 0 | 0 | 5.78e-16 | 0 | 0 |
| pe_distance | PE distance flag, flagged distance2 and gradient | real_kimono | 12741 | 0 | 4.34e-16 | 0 | 0 | 0 | 4.34e-16 | 0 | 0 |
| pt_distance | PT distance flag, flagged distance2 and gradient | degenerate | 2580 | 0 | 0.00e+00 | 0 | 0 | 0 | 0.00e+00 | 0 | 0 |
| pt_distance | PT distance flag, flagged distance2 and gradient | random | 4000 | 0 | 1.19e-15 | 0 | 0 | 0 | 1.19e-15 | 0 | 0 |
| pt_distance | PT distance flag, flagged distance2 and gradient | real_cloth_stack | 30000 | 0 | 6.07e-16 | 0 | 0 | 0 | 6.07e-16 | 0 | 0 |
| pt_distance | PT distance flag, flagged distance2 and gradient | real_kimono | 30000 | 0 | 1.58e-16 | 0 | 0 | 0 | 1.58e-16 | 0 | 0 |

Totals: cuda_fast-math-false: 989 failing rows of 1452423, cpu_fast-math-false: 989 failing rows of 1452423
