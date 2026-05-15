# Clinical Agentic AI

> Multi-agent system for clinical data derivation, verification, and traceability.

A regulated-pharma-shaped pipeline that takes a clinical dataset plus a derivation
specification and drives the work through a chain of cooperating agents — producing
an analysis-ready table and a complete, machine-readable audit trail. Everything is
designed for one property: a regulator can reconstruct any cell of the output from
the audit log alone.

---

## Run it in 60 seconds

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1            # macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_sample_data.py
cp .env.example .env                  # add ANTHROPIC_API_KEY to enable LLM
```

Two terminals from the project root:

```bash
# Terminal 1 — backend
uvicorn backend.main:app --reload --port 8000

# Terminal 2 — frontend
streamlit run frontend/app.py
```

Open `http://localhost:8501`, go to **New run**, upload a dataset and spec
(samples in `data/samples/` and `data/specs/`), confirm the preview, and submit.

---

## What to look at first (5-minute tour for a reviewer)

| File | Why it matters |
|---|---|
| [`backend/core/orchestrator.py`](backend/core/orchestrator.py) | The deterministic state machine. One file, 12 phases, every agent invocation explicit. Read this top-to-bottom to understand the full control flow. |
| [`backend/agents/`](backend/agents/) | One typed class per agent: SpecReviewer, DAGBuilder, Planner, CodeGenerator, CodePreApproval, StaticValidator, TestRunner, Executor, Verifier, Refiner, Auditor. Each obeys `run(state) -> state`. |
| [`backend/core/workflow_state.py`](backend/core/workflow_state.py) | The single dataclass that flows between agents; JSON-serialisable so every phase is checkpointed. |
| [`backend/core/sandbox.py`](backend/core/sandbox.py) | AST whitelist + namespace pinning. The first line of defence against untrusted code from the LLM. |
| [`backend/db/models.py`](backend/db/models.py) | Append-only audit substrate. Six tables, never `UPDATE`, only `INSERT`. |
| [`backend/memory/long_term.py`](backend/memory/long_term.py) | Two LTM stores: validated code patterns and resolved spec ambiguities. Both cross-run. |
| [`data/specs/sample_spec.yaml`](data/specs/sample_spec.yaml) | The bundled demo spec. One submission exercises every feature: multi-level DAG, ambiguity HITL, regulatory_critical preapproval HITL, test cases, custom invariants. |
| [`docs/design_document.md`](docs/design_document.md) | Architecture decisions, trade-offs, and production roadmap. |

---

## Architecture at a glance

```
INPUT: clinical CSV + derivation spec (YAML / JSON / CSV)

  1. Input Guardrails    : PII scan, schema check, type coercibility
  2. Spec Reviewer       : detect ambiguities    -> HITL: spec clarifications
  3. DAG Builder         : Kahn's topo sort, cycle detection
  4. Planner             : LLM produces per-derivation policy (risk, gates)
  5. Code Generator      : LTM hit -> reuse; else LLM with strict JSON contract
  6. Code Preapproval    : -> HITL: regulatory_critical or low-confidence review
  7. Static Validator    : AST whitelist on generated code
  8. Test Runner         : declared (input, expected) cases must pass
  9. Executor            : sandboxed per-row evaluation
 10. Verifier            : type / allowed_values / null rate / invariants
 11. Refiner             : bounded retries; -> HITL on exhaustion
 12. Auditor             : audit.json + audit.md, full hash chain

OUTPUT: analysis-ready table + hash-chained audit trail
```

Three first-class HITL gates (G1 clarifications, G2 code preapproval, G3 refiner
escalation). Role-gated UI: domain reviewers see plain-English forms, statistical
programmers see the raw code editor.

---

## Deliverables

- **Source code** — this repository.
- **Working prototype** — restart the two services above; open the Streamlit UI.
- **Design document** — `docs/design_document.docx` (architecture, trade-offs, production
  roadmap).
- **Tests** — `pytest tests/` runs the 32-test suite end-to-end including a full
  pipeline run against a stubbed LLM.

---

## Test the architecture

```bash
pytest tests/ -v
```

One command runs the full 37-test suite (~5 seconds). It covers:

| Suite | What it proves |
|---|---|
| [`tests/test_sandbox.py`](tests/test_sandbox.py) | The AST whitelist blocks imports, dunders, `open()`, exec/eval; safe code compiles; per-row execution captures exceptions |
| [`tests/test_dag.py`](tests/test_dag.py) | Kahn's topo-sort orders linear chains, alphabetises independent targets, and detects cycles |
| [`tests/test_guardrails.py`](tests/test_guardrails.py) | PII detector, schema mismatch, sentinel-spec guard, sandbox guard, out-of-domain block, null-rate warn |
| [`tests/test_memory.py`](tests/test_memory.py) | Pattern signature is stable; remember + lookup roundtrip works |
| [`tests/test_phases.py`](tests/test_phases.py) | Each of the 12 phases in isolation, plus the three HITL gates, planner risk-class routing, offline-mode escalation, confidence-driven HITL |
| [`tests/test_e2e.py`](tests/test_e2e.py) | One full pipeline run end-to-end, asserts the audit trail contents |
| [`tests/test_end_to_end_coverage.py`](tests/test_end_to_end_coverage.py) | **The five demo-grade scenarios:** happy path, admin code edit at G2 (override is persisted), Refiner repairs buggy code (out-of-domain → block → re-prompt → refined), Refiner exhaustion escalates to HITL, no-value-mapped detector blocks on missing branches |

The `test_end_to_end_coverage.py` file is the one to point an interviewer at:
each test is a labelled production-shaped scenario with assertions readable
without running the code.

---

## Docker

```bash
docker compose up --build
```

Boots the backend on `:8000` and frontend on `:8501` in two containers.

---

## Configuration

Every tunable lives in `.env` (see `.env.example`):

- `ANTHROPIC_API_KEY` — enable the LLM-backed code generator; without it the
  pipeline pauses at code generation and escalates to a manual-entry HITL gate.
- `OPENAI_API_KEY` - seemless Fallback to openAI LLM when there is an outage to Anthropic
- `REQUIRE_CODE_PREAPPROVAL` — when `true`, pause for every derivation rather than
  only those flagged by the Planner.
- `MIN_CONFIDENCE_THRESHOLD` — derivations whose self-reported LLM confidence falls
  below this number trigger HITL automatically.
- `PARALLEL_EXECUTOR` — flip to `true` to run independent same-level derivations
  concurrently; off by default for deterministic event ordering.
- `USER_EMAIL` / `USER_ROLES` — identity for the role-gated UI. In production these
  are injected by SSO; in development the in-app role switcher in the sidebar lets
  you flip between Reviewer and Admin without restarting.

---

## Project layout

```
backend/
  agents/         one file per agent
  api/routes/     FastAPI endpoints
  core/           orchestrator, sandbox, guardrails, llm_client, config
  db/             SQLAlchemy models, repositories, session
  eval/           golden-table evaluator
  memory/         short-term (workflow state) + long-term (patterns, clarifications)
  schemas/        Pydantic API schemas
  utils/          hashing, logging, console narrator, csv spec adapter
frontend/
  app.py          Streamlit entry point + navigation
  pages/          one file per page (Runs, New run, Reviews, Run detail, admin)
  components/     auth, journey mapper, stepper, status helpers
data/
  samples/        clinical_sample.csv  (30 patients)
  specs/          sample_spec.yaml / sample_spec.csv
  golden/         expected.csv  (for the evaluator)
docs/             design document
tests/            pytest suite
scripts/          one-off helpers (generate_sample_data.py)
docker/           Dockerfiles
```

---

## Troubleshooting

**Backend stuck / Streamlit says "Backend unreachable":** another `uvicorn` is
holding port 8000. Kill it:

```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique |
    ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }
```

**Run completes with stale code:** the long-term memory cache is serving patterns
from a previous prompt version. Clear it:

```powershell
Remove-Item storage\agentic.db,storage\memory,storage\checkpoints,storage\runs -Recurse -Force
```

**A reviewer says they cannot see the Admin pages:** that's by design. Set
`USER_ROLES=admin` in `.env` or switch the role in the sidebar (dev mode only).




