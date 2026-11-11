# IPC formula parity: cgq-contraction

Per case and input set: rows, then per backend the failing rows, the largest real error over the scale of its output (max |g - c| / s), and the rows that passed by the order clause and by the referee (compare.py).

| case | function | set | rows | fmad-true failures | fmad-true max err / s | fmad-true order rows | fmad-true referee rows |
|---|---|---|---|---|---|---|---|
| barrier | barrier, derivatives, normal force | degenerate | 4000 | 532 | 4.31e-05 | 0 | 0 |
| barrier | barrier, derivatives, normal force | random | 4000 | 0 | 1.03e-15 | 0 | 0 |
| barrier_ee | EE barrier gradient/Hessian (mollifier and flag dispatch) | degenerate | 5280 | 3 | 3.90e-12 | 0 | 0 |
| barrier_ee | EE barrier gradient/Hessian (mollifier and flag dispatch) | mollifier_sweep | 4000 | 0 | 1.71e-13 | 0 | 0 |
| barrier_ee | EE barrier gradient/Hessian (mollifier and flag dispatch) | parallel | 576 | 287 | 4.64e-02 | 0 | 0 |
| barrier_ee | EE barrier gradient/Hessian (mollifier and flag dispatch) | random | 4000 | 0 | 3.20e-13 | 0 | 0 |
| barrier_ee | EE barrier gradient/Hessian (mollifier and flag dispatch) | real_cloth_stack | 30000 | 0 | 7.59e-15 | 0 | 0 |
| barrier_ee | EE barrier gradient/Hessian (mollifier and flag dispatch) | real_cloth_stack_mollified | 18457 | 0 | 1.45e-14 | 0 | 0 |
| barrier_ee | EE barrier gradient/Hessian (mollifier and flag dispatch) | real_kimono | 30000 | 0 | 3.11e-17 | 0 | 0 |
| barrier_ee | EE barrier gradient/Hessian (mollifier and flag dispatch) | real_kimono_mollified | 514 | 0 | 1.59e-15 | 0 | 0 |
| barrier_pe | PE barrier gradient/Hessian (flag dispatch) | degenerate | 256 | 7 | 2.17e-11 | 0 | 0 |
| barrier_pe | PE barrier gradient/Hessian (flag dispatch) | random | 4000 | 0 | 3.38e-15 | 0 | 0 |
| barrier_pe | PE barrier gradient/Hessian (flag dispatch) | real_cloth_stack | 17883 | 0 | 1.87e-14 | 0 | 0 |
| barrier_pe | PE barrier gradient/Hessian (flag dispatch) | real_kimono | 12741 | 0 | 4.02e-16 | 0 | 0 |
| barrier_ph | PH barrier gradient/Hessian | degenerate | 1476 | 0 | 1.90e-16 | 0 | 0 |
| barrier_ph | PH barrier gradient/Hessian | random | 4000 | 0 | 2.99e-14 | 0 | 0 |
| barrier_pp | PP barrier gradient/Hessian | degenerate | 64 | 1 | 1.08e-11 | 0 | 0 |
| barrier_pp | PP barrier gradient/Hessian | random | 4000 | 0 | 4.19e-16 | 0 | 0 |
| barrier_pp | PP barrier gradient/Hessian | real_cloth_stack | 1404 | 0 | 7.11e-15 | 0 | 0 |
| barrier_pp | PP barrier gradient/Hessian | real_kimono | 882 | 0 | 2.33e-18 | 0 | 0 |
| barrier_pt | PT barrier gradient/Hessian (flag dispatch) | degenerate | 2048 | 12 | 1.27e-11 | 0 | 0 |
| barrier_pt | PT barrier gradient/Hessian (flag dispatch) | random | 4000 | 0 | 1.47e-13 | 0 | 0 |
| barrier_pt | PT barrier gradient/Hessian (flag dispatch) | real_cloth_stack | 30000 | 0 | 5.18e-15 | 0 | 0 |
| barrier_pt | PT barrier gradient/Hessian (flag dispatch) | real_kimono | 30000 | 0 | 4.01e-16 | 0 | 0 |
| candidate_ee | EE broad-phase distance rule | degenerate | 1280 | 0 | - | 0 | 0 |
| candidate_ee | EE broad-phase distance rule | random | 4000 | 0 | - | 0 | 0 |
| candidate_ee | EE broad-phase distance rule | real_cloth_stack | 30000 | 0 | - | 0 | 0 |
| candidate_ee | EE broad-phase distance rule | real_kimono | 30000 | 0 | - | 0 | 0 |
| candidate_pe | PE broad-phase distance rule | degenerate | 256 | 0 | - | 0 | 0 |
| candidate_pe | PE broad-phase distance rule | random | 4000 | 0 | - | 0 | 0 |
| candidate_ph | PH broad-phase cull | degenerate | 1476 | 0 | - | 0 | 0 |
| candidate_ph | PH broad-phase cull | random | 4000 | 0 | - | 0 | 0 |
| candidate_pp | PP broad-phase distance rule | degenerate | 64 | 0 | - | 0 | 0 |
| candidate_pp | PP broad-phase distance rule | random | 4000 | 0 | - | 0 | 0 |
| candidate_pt | PT broad-phase distance rule | degenerate | 2048 | 0 | - | 0 | 0 |
| candidate_pt | PT broad-phase distance rule | random | 4000 | 0 | - | 0 | 0 |
| candidate_pt | PT broad-phase distance rule | real_cloth_stack | 30000 | 0 | - | 0 | 0 |
| candidate_pt | PT broad-phase distance rule | real_kimono | 30000 | 0 | - | 0 | 0 |
| ccd_additive_ee | EE additive CCD | degenerate | 4560 | 59 | 1.64e-10 | 0 | 0 |
| ccd_additive_ee | EE additive CCD | random | 4000 | 0 | 5.35e-14 | 0 | 0 |
| ccd_additive_ee | EE additive CCD | real_cloth_stack | 30000 | 0 | 0.00e+00 | 0 | 0 |
| ccd_additive_ee | EE additive CCD | real_kimono | 30000 | 0 | 2.78e-15 | 0 | 0 |
| ccd_additive_pe | PE additive CCD | degenerate | 512 | 1 | 1.25e-12 | 0 | 0 |
| ccd_additive_pe | PE additive CCD | random | 4000 | 0 | 6.66e-16 | 0 | 0 |
| ccd_additive_pp | PP additive CCD | degenerate | 128 | 0 | 2.13e-14 | 0 | 0 |
| ccd_additive_pp | PP additive CCD | random | 4000 | 0 | 2.22e-16 | 0 | 0 |
| ccd_additive_pt | PT additive CCD | degenerate | 4096 | 4 | 4.41e-12 | 0 | 0 |
| ccd_additive_pt | PT additive CCD | random | 4000 | 0 | 8.32e-14 | 0 | 0 |
| ccd_additive_pt | PT additive CCD | real_cloth_stack | 30000 | 0 | 0.00e+00 | 0 | 0 |
| ccd_additive_pt | PT additive CCD | real_kimono | 30000 | 0 | 5.77e-15 | 0 | 0 |
| ccd_directional_ee | EE directional CCD | degenerate | 4560 | 329 | 1.15e-07 | 0 | 0 |
| ccd_directional_ee | EE directional CCD | random | 4000 | 9 | 2.51e-11 | 0 | 0 |
| ccd_directional_ee | EE directional CCD | real_cloth_stack | 30000 | 0 | 0.00e+00 | 0 | 0 |
| ccd_directional_ee | EE directional CCD | real_kimono | 30000 | 6 | 7.37e-12 | 0 | 0 |
| ccd_directional_pe | PE directional CCD | degenerate | 512 | 1 | 1.61e-12 | 0 | 0 |
| ccd_directional_pe | PE directional CCD | random | 4000 | 0 | 3.21e-14 | 0 | 0 |
| ccd_directional_pp | PP directional CCD | degenerate | 128 | 0 | 6.58e-14 | 0 | 0 |
| ccd_directional_pp | PP directional CCD | random | 4000 | 0 | 5.55e-16 | 0 | 0 |
| ccd_directional_pt | PT directional CCD | degenerate | 4096 | 7 | 2.00e-12 | 0 | 0 |
| ccd_directional_pt | PT directional CCD | random | 4000 | 1 | 3.92e-11 | 0 | 0 |
| ccd_directional_pt | PT directional CCD | real_cloth_stack | 30000 | 0 | 0.00e+00 | 0 | 0 |
| ccd_directional_pt | PT directional CCD | real_kimono | 30000 | 409 | 3.20e-01 | 0 | 0 |
| ccd_ph | PH CCD (linear) | degenerate | 1476 | 0 | 0.00e+00 | 0 | 0 |
| ccd_ph | PH CCD (linear) | random | 4000 | 0 | 6.78e-14 | 0 | 0 |
| ccd_ph_screw | PH screw CCD | degenerate | 1476 | 0 | 9.99e-16 | 0 | 0 |
| ccd_ph_screw | PH screw CCD | random | 4000 | 0 | 3.99e-14 | 0 | 0 |
| ccd_screw_ee | EE screw CCD (whole step) | degenerate | 1280 | 1 | 1.93e-01 | 0 | 0 |
| ccd_screw_ee | EE screw CCD (whole step) | random | 4000 | 32 | 4.02e-01 | 0 | 0 |
| ccd_screw_interval_ee | EE screw CCD (interval) | degenerate | 1280 | 0 | 1.25e-14 | 0 | 0 |
| ccd_screw_interval_ee | EE screw CCD (interval) | random | 4000 | 11 | 2.93e-01 | 0 | 0 |
| ccd_screw_interval_pt | PT screw CCD (interval) | degenerate | 2048 | 0 | 1.57e-14 | 0 | 0 |
| ccd_screw_interval_pt | PT screw CCD (interval) | random | 4000 | 9 | 2.69e-01 | 0 | 0 |
| ccd_screw_pe | PE screw CCD (whole step) | degenerate | 256 | 0 | 3.89e-15 | 0 | 0 |
| ccd_screw_pe | PE screw CCD (whole step) | random | 4000 | 0 | 6.72e-14 | 0 | 0 |
| ccd_screw_pp | PP screw CCD (whole step) | degenerate | 64 | 0 | 2.22e-16 | 0 | 0 |
| ccd_screw_pp | PP screw CCD (whole step) | random | 4000 | 0 | 8.88e-16 | 0 | 0 |
| ccd_screw_pt | PT screw CCD (whole step) | degenerate | 2048 | 4 | 2.51e-01 | 0 | 0 |
| ccd_screw_pt | PT screw CCD (whole step) | random | 4000 | 13 | 2.69e-01 | 0 | 0 |
| closest_ee | EE closest points | degenerate | 3632 | 597 | 1.61e-01 | 0 | 0 |
| closest_ee | EE closest points | random | 4000 | 0 | 3.04e-14 | 0 | 0 |
| closest_ee | EE closest points | real_cloth_stack | 30000 | 0 | 4.17e-15 | 0 | 0 |
| closest_ee | EE closest points | real_kimono | 30000 | 0 | 8.05e-16 | 0 | 0 |
| closest_pe | PE closest point | degenerate | 336 | 0 | 0.00e+00 | 0 | 0 |
| closest_pe | PE closest point | random | 4000 | 0 | 6.11e-16 | 0 | 0 |
| closest_pt | PT closest point | degenerate | 2580 | 0 | 0.00e+00 | 0 | 0 |
| closest_pt | PT closest point | random | 4000 | 0 | 9.44e-16 | 0 | 0 |
| closest_pt | PT closest point | real_cloth_stack | 30000 | 0 | 4.28e-15 | 0 | 0 |
| closest_pt | PT closest point | real_kimono | 30000 | 0 | 3.83e-16 | 0 | 0 |
| ee_distance | EE distance flag, flagged distance2 and gradient | degenerate | 3632 | 0 | 3.14e-13 | 0 | 0 |
| ee_distance | EE distance flag, flagged distance2 and gradient | random | 4000 | 0 | 2.71e-15 | 0 | 0 |
| ee_distance | EE distance flag, flagged distance2 and gradient | real_cloth_stack | 30000 | 0 | 1.90e-15 | 0 | 0 |
| ee_distance | EE distance flag, flagged distance2 and gradient | real_kimono | 30000 | 0 | 2.51e-16 | 0 | 0 |
| ee_distance_type | EE distance type (parallel projection) | degenerate | 3632 | 354 | - | 0 | 0 |
| ee_distance_type | EE distance type (parallel projection) | random | 4000 | 0 | - | 0 | 0 |
| ee_distance_type | EE distance type (parallel projection) | real_cloth_stack | 30000 | 4 | - | 0 | 0 |
| ee_distance_type | EE distance type (parallel projection) | real_kimono | 30000 | 0 | - | 0 | 0 |
| frame_ee | EE tangent basis and closest-point weights | degenerate | 868 | 0 | 3.00e-13 | 0 | 0 |
| frame_ee | EE tangent basis and closest-point weights | random | 3998 | 0 | 3.20e-14 | 0 | 0 |
| frame_ee | EE tangent basis and closest-point weights | real_cloth_stack | 19905 | 0 | 5.46e-15 | 0 | 0 |
| frame_ee | EE tangent basis and closest-point weights | real_kimono | 19902 | 0 | 3.86e-13 | 0 | 0 |
| frame_pe | PE tangent basis and closest-point weights | degenerate | 256 | 0 | 0.00e+00 | 0 | 0 |
| frame_pe | PE tangent basis and closest-point weights | random | 4000 | 0 | 1.11e-15 | 0 | 0 |
| frame_pe | PE tangent basis and closest-point weights | real_cloth_stack | 17883 | 0 | 2.55e-14 | 0 | 0 |
| frame_pe | PE tangent basis and closest-point weights | real_kimono | 12741 | 0 | 1.65e-14 | 0 | 0 |
| frame_ph | PH tangent basis (projector) | degenerate | 1476 | 0 | 0.00e+00 | 0 | 0 |
| frame_ph | PH tangent basis (projector) | random | 4000 | 0 | 4.44e-16 | 0 | 0 |
| frame_pp | PP tangent basis (projector) | degenerate | 64 | 0 | 5.55e-17 | 0 | 0 |
| frame_pp | PP tangent basis (projector) | random | 4000 | 0 | 5.55e-16 | 0 | 0 |
| frame_pp | PP tangent basis (projector) | real_cloth_stack | 1404 | 0 | 4.44e-16 | 0 | 0 |
| frame_pp | PP tangent basis (projector) | real_kimono | 882 | 0 | 4.44e-16 | 0 | 0 |
| frame_pt | PT tangent basis and closest-point weights | degenerate | 2048 | 0 | 0.00e+00 | 0 | 0 |
| frame_pt | PT tangent basis and closest-point weights | random | 4000 | 0 | 9.66e-14 | 0 | 0 |
| frame_pt | PT tangent basis and closest-point weights | real_cloth_stack | 20807 | 0 | 9.99e-16 | 0 | 0 |
| frame_pt | PT tangent basis and closest-point weights | real_kimono | 26473 | 0 | 1.67e-15 | 0 | 0 |
| friction_ee | EE friction energy/gradient/Hessian | degenerate | 868 | 0 | 1.08e-13 | 0 | 0 |
| friction_ee | EE friction energy/gradient/Hessian | random | 3998 | 0 | 9.69e-14 | 0 | 0 |
| friction_ee | EE friction energy/gradient/Hessian | real_cloth_stack | 19905 | 0 | 9.19e-15 | 0 | 0 |
| friction_ee | EE friction energy/gradient/Hessian | real_kimono | 19902 | 0 | 8.65e-15 | 0 | 0 |
| friction_pe | PE friction energy/gradient/Hessian | degenerate | 256 | 0 | 9.94e-17 | 0 | 0 |
| friction_pe | PE friction energy/gradient/Hessian | random | 4000 | 0 | 3.38e-15 | 0 | 0 |
| friction_pe | PE friction energy/gradient/Hessian | real_cloth_stack | 17883 | 0 | 1.57e-14 | 0 | 0 |
| friction_pe | PE friction energy/gradient/Hessian | real_kimono | 12741 | 0 | 8.08e-16 | 0 | 0 |
| friction_ph | PH friction energy/gradient/Hessian | degenerate | 1476 | 0 | 7.43e-17 | 0 | 0 |
| friction_ph | PH friction energy/gradient/Hessian | random | 4000 | 0 | 2.88e-13 | 0 | 0 |
| friction_pp | PP friction energy/gradient/Hessian | degenerate | 64 | 0 | 2.04e-16 | 0 | 0 |
| friction_pp | PP friction energy/gradient/Hessian | random | 4000 | 0 | 1.67e-15 | 0 | 0 |
| friction_pp | PP friction energy/gradient/Hessian | real_cloth_stack | 1404 | 0 | 1.06e-14 | 0 | 0 |
| friction_pp | PP friction energy/gradient/Hessian | real_kimono | 882 | 0 | 2.19e-16 | 0 | 0 |
| friction_pt | PT friction energy/gradient/Hessian | degenerate | 2048 | 0 | 9.86e-17 | 0 | 0 |
| friction_pt | PT friction energy/gradient/Hessian | random | 4000 | 0 | 3.30e-14 | 0 | 0 |
| friction_pt | PT friction energy/gradient/Hessian | real_cloth_stack | 20807 | 0 | 4.87e-15 | 0 | 0 |
| friction_pt | PT friction energy/gradient/Hessian | real_kimono | 26473 | 0 | 9.16e-15 | 0 | 0 |
| intersect | edge-triangle intersection | degenerate | 800 | 0 | - | 0 | 0 |
| intersect | edge-triangle intersection | random | 4000 | 0 | - | 0 | 0 |
| mollified_barrier | mollified barrier | degenerate | 4000 | 405 | 7.83e-07 | 0 | 0 |
| mollified_barrier | mollified barrier | random | 4000 | 0 | 1.81e-16 | 0 | 0 |
| mollifier | mollifier threshold and edge cross norm2 | degenerate | 3632 | 0 | 2.27e-16 | 0 | 0 |
| mollifier | mollifier threshold and edge cross norm2 | random | 4000 | 0 | 2.38e-16 | 0 | 0 |
| mollifier | mollifier threshold and edge cross norm2 | real_cloth_stack | 30000 | 0 | 2.48e-20 | 0 | 0 |
| mollifier | mollifier threshold and edge cross norm2 | real_kimono | 30000 | 0 | 4.84e-16 | 0 | 0 |
| pair_ee | EE pair d_hat, thickness, area weight | random | 4000 | 0 | 0.00e+00 | 0 | 0 |
| pair_pe | PE pair d_hat, thickness, area weight | random | 4000 | 0 | 0.00e+00 | 0 | 0 |
| pair_pp | PP pair d_hat, thickness, area weight | random | 4000 | 0 | 0.00e+00 | 0 | 0 |
| pair_pt | PT pair d_hat, thickness, area weight | random | 4000 | 0 | 0.00e+00 | 0 | 0 |
| pe_distance | PE distance flag, flagged distance2 and gradient | degenerate | 336 | 0 | 0.00e+00 | 0 | 0 |
| pe_distance | PE distance flag, flagged distance2 and gradient | random | 4000 | 0 | 4.22e-15 | 0 | 0 |
| pe_distance | PE distance flag, flagged distance2 and gradient | real_cloth_stack | 17883 | 0 | 2.08e-15 | 0 | 0 |
| pe_distance | PE distance flag, flagged distance2 and gradient | real_kimono | 12741 | 0 | 7.20e-15 | 0 | 0 |
| pt_distance | PT distance flag, flagged distance2 and gradient | degenerate | 2580 | 0 | 0.00e+00 | 0 | 0 |
| pt_distance | PT distance flag, flagged distance2 and gradient | random | 4000 | 0 | 3.02e-15 | 0 | 0 |
| pt_distance | PT distance flag, flagged distance2 and gradient | real_cloth_stack | 30000 | 0 | 1.92e-15 | 0 | 0 |
| pt_distance | PT distance flag, flagged distance2 and gradient | real_kimono | 30000 | 0 | 2.13e-16 | 0 | 0 |

Totals: fmad-true: 3098 failing rows of 1452423
