---
trigger: always_on
---

# EEG Workspace Rules

## Mission

This workspace is an active EEG Auditory Attention Decoding (AAD) research repository.

Preserve research correctness, reproducibility, and methodology over implementation speed.

---

## Mandatory Workflow

For every non-trivial implementation:

1. Understand the requested task.
2. Search the repository for related implementations before creating new code.
3. Search previous experiments and project memory for similar work.
4. Reuse existing implementations whenever possible.
5. Produce a short implementation plan before modifying code.
6. Only then implement.

---

## Before Finishing

Every code change must pass the autonomous engineering pipeline.

Never consider work complete until:

- Static validation passes.
- EEG validation passes.
- Autonomous review passes.
- Independent verification passes.

---

## Repository Rules

Do not duplicate existing models.

Do not introduce new architectures if an existing one can be extended.

Preserve repository structure.

Prefer modifying existing training pipelines rather than creating parallel versions.

---

## Research Rules

Treat methodological correctness as the highest priority.

Always look for:

- data leakage
- subject leakage
- validation leakage
- incorrect evaluation methodology
- reproducibility issues

If uncertain, investigate before implementing.

---

## External Research

When repository information is insufficient:

- Search official documentation.
- Search relevant papers.
- Search GitHub implementations.
- Compare approaches before implementing.

Do not blindly copy external code.

---

## Completion Criteria

A task is complete only after:

- implementation
- validation
- review
- verification

If any stage fails, continue improving the implementation until it passes or clearly explain why it cannot.