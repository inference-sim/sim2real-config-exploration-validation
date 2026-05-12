# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Experiment implementation for the Config Exploration section of the BLIS paper. Multiple LLM serving estimators search a shared config space, and their recommendations are validated against real llm-d deployments to measure sim2real drift and SLO compliance.

Part 1 is a comparative evaluation across all estimators. Part 2 is BLIS-only what-if analysis (SLO tiering, scaling curves, cross-model selection).

Paper plan: https://github.com/inference-sim/inference-sim/discussions/1237

## Estimators (git submodules)

All estimators live under `estimators/` as git submodules pinned to specific commits/tags. They exist so agents can read their source for understanding logic, not as build targets of this repo.

| Estimator | Summary Doc | Native Config Search |
|-----------|-------------|---------------------|
| inference-sim (Go) | [`estimators/INFERENCE-SIM.md`](estimators/INFERENCE-SIM.md) | No (single-point evaluator; use external sweep) |
| LLMServingSim (Python) | [`estimators/LLMSERVINGSIM.md`](estimators/LLMSERVINGSIM.md) | No (single-point evaluator; use external sweep) |
| AIConfigurator (Python) | [`estimators/AICONFIGURATOR.md`](estimators/AICONFIGURATOR.md) | Yes (`cli default` sweeps configs against SLA targets) |
| Vidur (Python) | [`estimators/VIDUR.md`](estimators/VIDUR.md) | Yes (`config_optimizer` with binary search under SLO constraints) |
| llm-optimizer (Python) | [`estimators/LLM-OPTIMIZER.md`](estimators/LLM-OPTIMIZER.md) | Yes (grid search with `--constraints` for SLO filtering) |

Each summary doc covers installation, CLI usage, arguments, and config search capabilities. Refer to these when determining how to invoke or integrate each tool.

For a side-by-side comparison of parameters, metrics, and search mechanisms across all estimators, see [`estimators/COMPARISON.md`](estimators/COMPARISON.md).

To initialize submodules after clone: `git submodule update --init --recursive`

## Dependencies

Python dependencies: `pip install -r requirements.txt`

## Rules

- Never use em-dashes (the "--" or Unicode U+2014 character) anywhere in this repository, including documentation, comments, and commit messages. Use commas, semicolons, parentheses, or separate sentences instead.
- When changing estimator versions or adding new ones, always update the Estimators table in README.md with the correct version/commit.
- When you need to explore or understand details about an estimator (reading its source code, checking CLI arguments, understanding internal logic), always spawn a subagent to do that research. Never read estimator source files directly in the main session unless the user explicitly asks you to. The summary docs (estimators/*.md) and COMPARISON.md can be read in the main session; the submodule source code under estimators/*/ must not be. Use the `ask-estimator` skill for questions about estimators; it handles subagent dispatch automatically.
