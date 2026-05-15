# Design Document — Clinical Agentic AI

**Audience:** Reviewers for the Sanofi Digital R&D AI/ML Lead candidate assignment.
**Scope:** A working multi-agent prototype that turns a clinical dataset and a
derivation specification into an analysis-ready table with a full audit trail,
suitable for regulated work.
**Length:** Approximately four pages.

---

## 1. Problem framing

Clinical data workflows take structured input datasets and transform them into
analysis-ready outputs that drive reporting, regulatory submission, and clinical
decision-making. The work has to balance four pressures that are often in tension:

- **Automation.** A study can have hundreds of derived variables; manual coding
  does not scale.
- **Correctness.** Each cell must be defensible against an auditor.
- **Auditability.** Reproducing any output value from its inputs is non-negotiable.
- **Human oversight.** Where the system is uncertain or the rule is high-stakes,
  a domain expert has to remain in the loop.

A naive solution — "ask an LLM to write the code" — fails on at least three of
those pressures. The system in this repository is the alternative: a deterministic
orchestrator drives a fixed sequence of typed agents, each of which has *bounded*
autonomy. The LLM is a worker, not the controller.

---

## 2. Architecture

### 2.1 Pipeline shape

The orchestrator advances through twelve phases. Three of them are explicit
human-in-the-loop gates; the rest run automatically with safety guards.

```
INPUT: clinical CSV + derivation spec (YAML, JSON, or CSV)

  1. Input Guardrails    : PII scan, schema check, type coercibility
  2. Spec Reviewer       : detect ambiguities; HITL gate G1 on hits
  3. DAG Builder         : Kahn topological sort + cycle detection
  4. Planner             : LLM produces per-derivation policy
  5. Code Generator      : LTM hit -> reuse; otherwise LLM with strict JSON contract
  6. Code Preapproval    : HITL gate G2 (regulatory_critical OR low confidence)
  7. Static Validator    : AST whitelist on generated code
  8. Test Runner         : declared (input, expected) cases must pass
  9. Executor            : sandboxed per-row evaluation in topological order
 10. Verifier            : type / allowed_values / null rate / custom invariants
 11. Refiner             : bounded retries; HITL gate G3 on exhaustion
 12. Auditor             : audit.json + audit.md, full hash chain

OUTPUT: analysis-ready table + audit trail
```

### 2.2 Why hand-rolled orchestration

I chose a deterministic finite-state machine instead of a graph framework
(LangGraph, CrewAI). The reasoning:

- **Transparency.** An auditor reads one Python file
  (`backend/core/orchestrator.py`) and understands the full control flow. A
  graph framework adds implicit semantics the audit defence would have to learn.
- **First-class HITL pause.** The orchestrator can halt mid-pipeline, persist
  state to disk, and re-enter from any checkpoint after an arbitrary delay.
  Frameworks that treat human input as just another node make this harder.
- **Bounded agentic behaviour.** Agents have autonomy *inside* their step (LLM
  call, memory lookup, retry, escalation) but not over *which* step runs next.
  The combination gives audit-friendly determinism at the seam level and
  intelligent worker behaviour at the leaf level.

### 2.3 Agents and their responsibilities

Each agent is a typed class with one public method (`run(state) -> state`) and a
single job:

| Agent | Decides | Hard contract |
|---|---|---|
| Spec Reviewer | Is this rule unambiguous? | Surface clarifications, never guess silently |
| DAG Builder | What is the execution order? | Reject cycles or unknown columns |
| Planner | How critical is each derivation? | Per-derivation policy with hash; may only escalate risk |
| Code Generator | Memory hit, LLM call, or escalation? | Code must pass static check; reasoning captured |
| Code Preapproval | Does this batch need a human? | Bundle into a single HITL request with previews |
| Static Validator | Is this code safe to run? | AST whitelist; no imports, IO, dunders |
| Test Runner | Does this code pass declared examples? | Block before full execution if any case fails |
| Executor | Run the code on the dataset | Per-row error capture; never crashes the run |
| Verifier | Does the output respect the contract? | Block on out-of-domain or egregious null rate |
| Refiner | Can the failure be fixed automatically? | Bounded retries (default 3), then escalate |
| Auditor | Can a regulator reconstruct this run? | Produce hash-chained audit.json + audit.md |

---

## 3. Dependency handling

Derived variables can consume raw columns and other derived columns. The DAG
Builder constructs a directed graph from `sources` references, runs Kahn's
algorithm with alphabetical tiebreak (deterministic order), and rejects cycles
or references to columns the spec did not declare and that are not another
derivation.

The Executor walks targets in topological order, materialising each derived
column into a shared DataFrame so subsequent functions see upstream outputs
through the same `row["name"]` interface as raw inputs. The Code Generator's
prompt explicitly annotates each source as RAW (raw input column with declared
type) or DERIVED (produced by an earlier derivation in this spec, with its
upstream rule shown) so the LLM can pick the right coercion (`to_int` /
`to_float`) for derived values.

Multi-level dependency is exercised by the bundled sample: `ANALYSIS_POP_FLAG`
depends on `TREATMENT_DURATION`; `COMPOSITE_RISK_TIER` depends on three
Level-2 derivations.

---

## 4. Human-in-the-Loop design

### 4.1 Three gates, each purpose-built

| Gate | Trigger | Reviewer interaction |
|---|---|---|
| **G1 Spec clarifications** | Spec Reviewer flags ambiguity | Per-question text fields; each answer is folded into the corresponding `rule` text before code generation; new `resolved_spec_hash` recorded |
| **G2 Code preapproval** | Planner forced (regulatory_critical) or LLM confidence below threshold | Per-derivation card: rule + reasoning + dry-run preview on first 3 rows; sub-actions approve / edit / regenerate-with-hint |
| **G3 Refiner escalation** | Verifier blocks and Refiner exhausts retry budget | Failure context with code + recent findings; reviewer edits Python directly (admin role) or asks for changes in English (T1 role) |

### 4.2 Capture model

Every decision becomes an immutable row in `hitl_decisions`: reviewer, action,
structured payload (`clarification_answers` map or `derivation_overrides` map),
timestamp. A parallel row in `audit_entries` records actor, action, and the
run-state hash at decision time. Both tables are append-only — corrections
create new rows, never updates.

### 4.3 Feedback loops

- **Within-run:** clarification answers fold into the rule; the Code Generator
  reads the updated rule on the next phase. Per-derivation overrides at the
  preapproval gate either approve, replace the code, or trigger a regeneration
  with the reviewer's plain-English hint woven into the next LLM prompt.
- **Cross-run:** answered clarifications are written to `clarification_memory`,
  keyed by `(target, issue)` signature. The next run that raises a similar
  ambiguity sees the previous answer pre-filled.
- **System level:** the HITL Quality dashboard surfaces intervention rate,
  false-alert rate, memory reuse rate, and decision latency so the review
  process itself can be measured and tuned over time.

### 4.4 Role-gated UI

Domain reviewers (`USER_ROLES=user`) see only safe choices: approve, request
changes in plain English, reject. Statistical programmers and senior reviewers
(`admin`) additionally see the raw code editor. The same backend route handles
both — the difference is purely UI, with server-side enforcement available
behind a `require_admin` dependency for the equivalent production endpoint.

---

## 5. Traceability and reproducibility

The audit substrate is six append-only tables: `runs`, `agent_events`,
`derivations`, `validations`, `hitl_decisions`, `audit_entries`, plus two
long-term memory tables (`memory_patterns`, `clarification_memory`). Every row
points back at its inputs through stable hashes.

The full hash chain:

```
spec_hash  ->  resolved_spec_hash  ->  plan_hash  ->  code_hash[per target]  ->  output_hash
```

Given any output value, an auditor can:

1. Find the row in `derivations` and read its `code_hash`.
2. Read the verbatim code (stored on the row).
3. Read the validator outcomes for that target from `validations`.
4. Read any human decisions for that target from `hitl_decisions`.
5. Reconstruct the spec version from `resolved_spec_hash`.
6. Replay the run from the JSON checkpoints in `storage/checkpoints/<run_id>/`.

Two artefact files are produced at the end of every run: a machine-readable
`audit.json` (for downstream tooling) and a human-readable `audit.md` (for the
regulator's eyes). Both are downloadable from the UI.

---

## 6. Memory and reusability

### 6.1 Short-term — workflow state

The `WorkflowState` dataclass flows through every agent. Every field is
JSON-serialisable; the orchestrator checkpoints state to disk after each phase
so a crash or HITL pause is recoverable. The same state is mirrored into the
audit DB for human inspection. STM is discardable after a run completes; only
LTM persists.

### 6.2 Long-term — two stores

**Pattern memory** (`memory_patterns`) caches validated `derive(row)` functions
keyed on `(target_name, sorted(sources), tokenised(rule), prompt_version)`.
Including `prompt_version` invalidates cached patterns automatically when the
Code Generator's prompt is upgraded — preventing prompt regressions from being
hidden by stale memory hits.

**Clarification memory** (`clarification_memory`) caches reviewer answers keyed
on `(target_name, tokenised(issue))`. When a future run raises a similar
ambiguity the Spec Reviewer pre-fills the form with the stored answer, which
the reviewer can one-click accept or override.

### 6.3 What the cache earns the system

- **Cost.** Memory hits skip the LLM call entirely.
- **Determinism.** A reviewer-edited code snippet becomes the canonical version
  across all subsequent studies.
- **Consistency.** The same rule across two studies produces identical code
  (with the same hash), which is a non-trivial regulatory property.

---

## 7. Verification and reliability

Five defensive layers, applied in order:

| Layer | Catches |
|---|---|
| Input Guardrails | Missing columns, PII columns, unparseable types |
| Static Validator | Imports, IO, dunders, attribute access on `row`, syntax errors |
| Test Runner | Deterministic bugs detectable from spec-declared examples |
| Verifier | Wrong type, out-of-domain category, egregious null rate, custom invariant failure |
| Refiner | Failures the LLM can plausibly self-repair; escalates otherwise |

A null rate more than 2x the declared `max_null_rate` (or any null when the cap
is zero) is treated as an egregious overshoot and BLOCKS rather than warns,
forcing the Refiner to attempt a fix. The Verifier also sets the derivation's
status to `failed` when it blocks, so the Refiner picks it up on the next loop
iteration. Stale validations are cleared at the top of each Verifier pass so the
loop sees only the latest findings.

---

## 8. Trade-offs I would defend

| Decision | Alternative | Why I chose this |
|---|---|---|
| Hand-rolled FSM | LangGraph / CrewAI / AutoGen | One Python file, fully transparent to an auditor |
| LLM mandatory, no hardcoded fallback library | Generic rule library that handles common shapes | A hardcoded library masquerading as a fallback gives false confidence on rules it was never written for; "LLM mandatory, otherwise HITL" is the honest contract |
| Per-question HITL form | Single approve / reject button with freeform comment | Answers become structured data — queryable, prefillable from memory, replayable |
| Test-case gate before full execute | Verifier only, post-execute | Catches errors in milliseconds on three rows instead of seconds on N rows; author-written test cases are the most valuable form of validation |
| Append-only audit DB | Mutable state with updates | Tamper evidence and reproducibility; the slight storage overhead is irrelevant compared to the audit gain |
| Signature-based LTM (exact rule tokens) | Vector-embedded semantic similarity | Deterministic, explainable, no embedding-model dependency for retrieval. Vector recall is a v2 add behind the same interface |
| Daemon-thread orchestrator | Async / Celery worker | One-process simplicity for a prototype; the production system would absolutely swap in a real worker queue (called out below) |

---

## 9. What works today, what is design-only

### 9.1 Built and tested in the prototype

- 12-phase deterministic orchestrator with checkpoints
- 8 typed agents (11 if you count Code Preapproval, Test Runner, Static
  Validator as distinct phases)
- 3 HITL gates with structured forms and role-gated UI
- DAG Builder with deterministic topological order
- AST sandbox with namespace pinning
- Test Runner with spec-declared examples
- Verifier with 5 finding categories
- Refiner with bounded retries and HITL escalation
- 2 LTM stores with prompt-version invalidation
- Append-only audit DB + audit.json / audit.md
- CSV + YAML spec adapters
- Auto-inferred source schema when missing
- Background-threaded execution with frontend polling
- Resume-from-checkpoint endpoint and UI affordance
- Live status badge with current phase, global review banner, toast notifications
- Download artefacts (CSV / Parquet / audit.json / audit.md) from the UI
- 32-test pytest suite covering each phase
- GitHub Actions CI workflow

### 9.2 Production roadmap (design-only)

These belong in a real deployment but are out of scope for a one-week prototype:

| Item | What it adds |
|---|---|
| **G4 output sign-off gate** | A final HITL gate before the output leaves the system, with a cryptographic e-signature recorded against the audit trail (21 CFR Part 11 §11.50) |
| **WORM audit bucket** | S3 Object Lock or equivalent for tamper-evident audit storage |
| **Multi-table relational input** | Real pharma data is SDTM with 10+ joined tables; today the system takes a single CSV |
| **Per-patient tier of derivations** | Aggregations like "last observation carried forward" or "patient's worst lab" need a tier above per-row |
| **Reference-data plugin** | MedDRA / WHODrug code lookups with version pinning in the audit trail |
| **LLM gateway with circuit breaker** | Private Bedrock / Vertex endpoint, per-tenant rate limiting, cost tracking, failover model selection |
| **OpenTelemetry traces** | Per-phase spans pipelined to a central observability stack |
| **Drift detection** | Periodic comparison of output distributions across runs |
| **Streaming execution** | Chunked per-row processing for datasets that do not fit in memory |
| **Multi-tenant isolation** | Tenant ID on every row, row-level security policies, per-tenant KMS keys |

### 9.3 Honest assessment of failure modes

The architecture catches every failure mode I designed for: ambiguous specs,
unsafe code, deterministic bugs, type / domain violations, excessive null rates,
silent code crashes, backend death mid-run, CSV format confusion (dataset
uploaded as spec). The single class it does not catch on its own is
**plausible-but-wrong** code — code that passes the static validator, the
test-cases gate, and the verifier but produces logically wrong values on real
data (e.g., a subtle off-by-one in a threshold). The right defence is either
richer test cases (which is the spec author's responsibility) or a golden-table
comparison wired into CI. The Evaluation page exists for manual golden
comparison; making it a CI gate is the next step.

---

## 10. Mapping to the assessment criteria

| Criterion | Where evidence lives |
|---|---|
| 1. Agentic architecture | 12-phase pipeline; typed agents; one-file orchestrator; bounded autonomy |
| 2. Data logic and dependency | DAG with topological sort; multi-level dependency demonstrated by sample; auto-inferred schema |
| 3. Verification and reliability | 5 defensive layers; bounded refiner; egregious null-rate BLOCK; verifier sets failure status to drive refiner |
| 4. HITL design | 3 structured gates; per-question and per-derivation forms; role-gated UI; memory pre-fill |
| 5. Traceability and auditability | Hash chain across spec / plan / code / output; append-only DB; audit.json + audit.md; downloads from UI |
| 6. Memory and reusability | STM checkpoints + audit; LTM patterns + clarifications; prompt-version invalidation |
| 7. Implementation quality | 32 passing pytest; modular layout; CI workflow; Docker compose |
| 8. Communication and reasoning | This document, README's "what to look at first", live UI demo, presentation slides |
