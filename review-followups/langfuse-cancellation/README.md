# Cross-consumer cancellation regression

Review artifacts for [PowerContext #2517](https://github.com/oceanbase/powercontext/pull/2517), based on `a408ddecc747548030a14e755066d882b9623214`. These files are not proposed PowerContext runtime changes or part of its default test suite.

The strict regression executes official `create_agent`, the actual PowerContext HTTP SDK and ASGI/SQLite server, the real Langfuse default export filter, and a deterministic asynchronous model. A started model body must have an ended generation. The fixture covers normal completion, a business exception, tool rounds, cancellation, stream cancellation/close, and concurrent success/cancellation. It does not invent a generation, response, token usage, or cost.

## Ownership and exact versions

- PowerContext core and its matching LangChain adapter: source head above.
- LangChain 1.4.4 / `langchain-core` 1.7.6; core source tag commit `b4aa7e7d00c452251720ce30b8bc2c41ca44e1f5`.
- Langfuse Python SDK 5.0.2; source tag commit `35f8e7ee0c9c450cecf059563edbab14656a9fd1`.
- The application owns its callback instance (`run_inline=True`) and a public per-invocation OpenTelemetry `use_span` boundary. This preserves normal phase parentage and caller context; it does not replace either consumer lifecycle correction.

`langchain-core-1.7.6-cancellation.patch` belongs to LangChain's model-owning gather frame. It delivers terminal callbacks for that frame's finished tasks when the outer gather is cancelled, without scanning arbitrary consumer ancestry or terminating unrelated generations.

`langfuse5-cancel-generation-fix.patch` belongs to Langfuse. It retains explicit cancellation type/reason, leaves unreported cancellation cost unknown, and clears first-token memo state. It does not change global callback dispatch defaults.

## Run the strict reproduction

Start from this PowerContext checkout with its matching adapter and the versions above installed, including the optional Langfuse and OTLP HTTP exporter test dependencies. Local verification used Python 3.14.3. Missing optional dependencies can skip collection; that is not a passing lifecycle result.

From the PowerContext repository root, run against the unmodified official consumer packages:

```sh
UV_OFFLINE=1 UV_CACHE_DIR=/tmp/powercontext-pctx-audit-uv-cache \
PYTHONPATH="$PWD/src:$PWD/integrations/langchain/src" \
POWERCONTEXT_LIFECYCLE_EVIDENCE_DIR= \
  .venv/bin/python -m pytest -q review-followups/langfuse-cancellation/test_callback_cancellation.py
```

Expected baseline result: a strict failure because the cancelled model body starts but its generation is absent (`0 != 1`). This is a reproduced external dependency gap, not a green PowerContext gate.

Apply each included patch in the owning source checkout at the exact tag commit above. For the patched run, set `LANGCHAIN_CORE_REVIEW_DIR` to the patched checkout's `libs/core` directory and `LANGFUSE_REVIEW_DIR` to the patched SDK repository root. Then run:

```sh
UV_OFFLINE=1 UV_CACHE_DIR=/tmp/powercontext-pctx-audit-uv-cache \
PYTHONPATH="$LANGCHAIN_CORE_REVIEW_DIR:$LANGFUSE_REVIEW_DIR:$PWD/src:$PWD/integrations/langchain/src" \
POWERCONTEXT_LIFECYCLE_EVIDENCE_DIR= \
  .venv/bin/python -m pytest -q review-followups/langfuse-cancellation/test_callback_cancellation.py
```

Expected patched result: one passing strict lifecycle regression. Assertions are unchanged between baseline and patched runs. Both commands were verified locally with the source directories supplied for the second run.

The preserved multi-scenario baseline recorded 182 spans, 10 started model bodies, 6 exported generations, and 4 residual callback observations/tokens. With both reviewed consumer changes, it recorded 186 spans, 10 bodies, 10 generations, and zero residual observations/tokens. Fixed model outputs are lifecycle fixtures, not inference-quality evidence.

For a fresh OTLP capture, set `POWERCONTEXT_LIFECYCLE_EVIDENCE_DIR` to a new empty evidence directory. Never overwrite preserved evidence or commit raw trace payloads. The sink is in memory; no external model or deployed collector is needed.

Please track and review the two owning consumer patches independently. Do not vendor them into PowerContext or add this strict cancellation expectation to its default green suite until the supported consumer lifecycle is resolved.

Raw unified-diff files retain significant context-line whitespace. Source/document whitespace checks exclude those patch artifacts; their bytes are preserved in `artifact-hashes.json`.
