# Agent Instructions

This checkout develops the graph-native Rigid Newton coupling runtime.

Before editing code, read:

1. [Genesis development guidelines](CLAUDE.md)
2. [Rigid Newton roadmap](dev-docs/rigid-newton-roadmap.md)
3. [Rigid Newton development guide](dev-docs/rigid-newton-development.md)

The roadmap is authoritative for requirements and milestone order. Do not
decide an item marked **Open** without user approval. Keep integration behind
`NewtonCoupler`, do not add an eager solver path, and document every upstream
workaround with an issue, regression test, measured cost, and explicit
approval.
