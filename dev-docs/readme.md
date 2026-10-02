# Rigid Newton Development Documents

Read these documents before changing the Rigid Newton framework:

1. [Roadmap](rigid-newton-roadmap.md)
2. [Development guide](rigid-newton-development.md)
3. [QIPC simulation system design](qipc-simulation-system-design.md)
4. [CGQ contact parameter manifest](cgq-contact-parameter-manifest.md)
5. [Contact performance debt register](contact-performance-debt.md)
6. [Reduced-KKT contact proxy migration](reduced-kkt-migration.md)
7. [CGQ MinCoo rigid dynamics backend](cgq-mincoo-rigid-backend.md)
8. [Quadrants compile-speed optimization handoff](quadrants-compile-speed-optimization-handoff.md)
9. [Static graph composition and data-oriented rationale](data-oriented-static-dispatch-rationale.md)
10. [Component-partitioned PCG](partition-pcg.md)

Hard admission gate: production and milestone runtime code must be
load-balanced and must use the most efficient applicable warp/subgroup-level
algorithm. Non-load-balanced or non-warp-optimal scene-scale code is forbidden;
diagnostic references must remain isolated from builders and examples.
