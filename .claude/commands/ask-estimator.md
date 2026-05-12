Answer a technical question about one of the estimators in this repo.

## Usage

/ask-estimator <estimator-name> <question>

## Estimators

- LLMServingSim → summary: `estimators/LLMSERVINGSIM.md`, source: `estimators/LLMServingSim/`
- aiconfigurator → summary: `estimators/AICONFIGURATOR.md`, source: `estimators/aiconfigurator/`
- vidur → summary: `estimators/VIDUR.md`, source: `estimators/vidur/`
- llm-optimizer → summary: `estimators/LLM-OPTIMIZER.md`, source: `estimators/llm-optimizer/`
- inference-sim → summary: `estimators/INFERENCE-SIM.md`, source: `estimators/inference-sim/`

Cross-estimator comparison: `estimators/COMPARISON.md`

## Procedure

1. Identify which estimator the user is asking about from the arguments.
2. Read the corresponding summary document for high-level context.
3. If the summary fully answers the question, respond directly. Then skip to step 6.
4. If the summary is insufficient, spawn a subagent (using the Agent tool) to explore the estimator's source directory. The subagent should grep/find/read the source and return findings with file paths and line numbers.
5. Synthesize the subagent's findings into a concise answer for the user.
6. **Update the summary document.** After answering, always update the estimator's summary doc with any information that was discovered during this interaction but was missing or incomplete in the doc. This includes new CLI flags, undocumented behaviors, corrected descriptions, additional architecture details, or anything else that would help future readers. If no new information was discovered (the summary already covered everything), skip this step.
7. **Update COMPARISON.md if applicable.** If the new information affects a dimension already tracked in `estimators/COMPARISON.md` (parameters, metrics, search mechanisms, supported hardware, etc.), or reveals a new dimension worth comparing, update COMPARISON.md as well. Read it first to understand the existing structure before editing.

## Update guidelines

- Integrate new information into the existing document structure. Add to relevant sections rather than appending a raw "Notes from Q&A" block.
- Preserve the document's tone and formatting conventions.
- If a section needs to be created (the topic is genuinely new), place it logically relative to existing sections.
- Do not remove or rewrite correct existing content; only add, correct, or clarify.
- Keep the summary docs authoritative: every factual claim should be traceable to source code. Include file paths and relevant details where helpful.

## Notes

- Always start with the summary doc; it contains API surface, architecture, and key concepts.
- Never read estimator source files (under estimators/*/) directly in the main session. Always delegate source exploration to a subagent.
- If the estimator name is ambiguous or missing, ask which one.
- Always provide code proofs: cite specific file paths, line numbers, and relevant code snippets when answering from source. Never make claims about how an estimator works without pointing to the code that proves it.
