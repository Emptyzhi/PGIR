# WildClawBench Integration Overlay

This directory contains PGIR-owned files that overlay an official
WildClawBench checkout. It does not redistribute benchmark tasks, workspaces,
Docker images, graders, or third-party agent code.

Copy `src/` and `tests/` into the corresponding WildClawBench directories.
The integration adds:

- runtime contracts and automatic contract synthesis;
- provenance tracking and minimal repair-frontier selection;
- snapshot, restore, replay, and global-replan adapters;
- PGIR, no-provenance, no-contract, and local-leaf runtime conditions.

All conditions share the same underlying OpenClaw execution surface.
