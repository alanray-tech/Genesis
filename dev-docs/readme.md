# Rigid Newton Development Documents

Read these documents before changing the Rigid Newton framework:

1. [Roadmap](rigid-newton-roadmap.md) — current product scope and accepted decisions.
2. [Development guide](rigid-newton-development.md) — current architecture, workflow, and verification.
3. [Graph-native simulation system design](graph-native-simulation-system-design.md) — system and lifecycle contracts.
4. [Contact parameter manifest](contact-parameter-manifest.md) — internal contact configuration contract.
5. [Reduced-KKT contact proxy contract](reduced-kkt-contact-proxy.md) — proxy ownership and algebra.
6. [Contact performance debt register](contact-performance-debt.md) — current measured optimization queue.

Files under `performance-report/` are historical measurement evidence. They do
not override the current scope or runtime contracts above.

Hard admission gate: production and milestone runtime code must be
load-balanced and must use the most efficient applicable warp/subgroup-level
algorithm. Non-load-balanced or non-warp-optimal scene-scale code is forbidden;
diagnostic references must remain isolated from builders and examples.
