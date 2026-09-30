# Genesis World QIPC Coupling Performance Report

This release packages the measured runtime and compilation optimizations made
while migrating CGQ cloth, IPC contact, reduced KKT, and GPU graph execution
into Genesis World.

Highlights:

- strict runtime measurements for contact, BVH, sort/reduce, reduced-KKT, and
  Quadrants scheduling/lowering optimizations;
- cold-compilation analysis and the Quadrants pass-level fixes;
- 20-warmup + 250-measured-frame validation across four pure-cloth workloads;
- final interpretable Genesis / CGQ normalized ratios between `0.965x` and
  `1.005x`, with the trajectory-divergent stress case reported separately;
- reproducible source data and plot-generation script.

Assets:

- `qipc-coupling-performance-report.md`: complete experimental report;
- `qipc-performance-optimization-summary.png`: visual optimization summary;
- `qipc-coupling-performance-data.json`: machine-readable measurement ledger;
- `genesis-qipc-performance-report-2026-09-30.zip`: bundled report, plot, data,
  and generator.

The reported speedups are not additive. The report distinguishes strict
same-window A/B evidence, pinned CGQ migration evidence, numerical-work
corrections, and trajectory-sensitive aggregates.

The referenced Quadrants commits are published on
`alanray-tech/quadrants:dev/compile-time-opt`. JSON/CSV summaries are included
or referenced; the report identifies where an original Nsight SQLite capture
must be regenerated from the documented profile window.
