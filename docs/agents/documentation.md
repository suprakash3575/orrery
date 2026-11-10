# Documentation expectations

Read this before writing docs, reports, or learner-facing diagnostics. It
extends [AGENTS.md](../../AGENTS.md).

Update documentation when externally visible behavior changes:

- `README.md` for current supported behavior and commands.
- `PLAN.md` when an implementation step is completed or materially revised.

Comments should explain intent, field meaning, lifecycle, rule ordering,
mathematical reasoning, and edge cases. Avoid comments that merely repeat the
syntax of the following line.

For learner-facing diagnostics and other multi-stage teaching functions, add
plain-language comments before each conceptual stage. Explain what form the
data is in, which boundary it crosses, why the stage is needed, and what it does
not do; do not assume that the reader can infer the lifecycle from function
names alone. When a trace contains normalized or encoded values, present the
real domain value and event first, then show the model representation as
supporting detail; do not make readers reconstruct gameplay from raw feature
diffs.
