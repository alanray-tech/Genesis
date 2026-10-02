# Coupling Performance Report

Status: historical release artifact, not a current runtime contract.

This release packages the measured runtime and compilation optimizations made
while migrating cloth, contact, reduced KKT, and GPU graph execution into the
current runtime.

Highlights:

- strict runtime measurements for contact, BVH, sort/reduce, reduced-KKT, and
  Quadrants scheduling/lowering optimizations;
- cold-compilation analysis and the Quadrants pass-level fixes;
- 20-warmup + 250-measured-frame validation across four pure-cloth workloads;
- final interpretable runtime/reference normalized ratios between `0.965x` and
  `1.005x`, with the trajectory-divergent stress case reported separately;
- reproducible source data and plot-generation script.

Assets:

- `coupling-performance-report.md`: complete experimental report;
- `coupling-performance-summary.png`: visual optimization summary;
- `coupling-performance-data.json`: machine-readable measurement ledger;
- `coupling-performance-report-2026-09-30.zip`: bundled report, plot, data,
  and generator.

The reported speedups are not additive. The report distinguishes strict
same-window A/B evidence, pinned-reference migration evidence, numerical-work
corrections, and trajectory-sensitive aggregates.

The machine-readable ledger records every tested compiler commit. JSON/CSV
summaries are included or referenced; the report identifies where an original
Nsight SQLite capture must be regenerated from the documented profile window.
