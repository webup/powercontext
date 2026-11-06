# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Actual runtime coverage for application-owned async callback configuration.

Cancellation requires the model-owning LangChain frame to deliver its terminal
callback and the consumer to export that interrupted generation without invented
usage. This assertion deliberately fails with unpatched langchain-core 1.7.6 and
Langfuse 5.0.2; local verification combines their two targeted consumer patches.
"""

import asyncio
import base64
import hashlib
import inspect
import json
import os
import tempfile
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import uuid4

import httpx
import pytest
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import tool
from opentelemetry import context, trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from powercontext_langchain import PowerContextMiddleware, PowerContextScope
from powercontext_langchain.client import shared_http_client

from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.runtime import InferenceConfig, RuntimeConfig
from powercontext.client import PowerContextClient
from powercontext.http import RememberMemoryRequest, ResolveScopeBindingRequest
from powercontext.server.factory import create_server_app
from powercontext.server.settings import McpConfig, ServerSettings
from powercontext.server.tracing import ServerTracing

langfuse = pytest.importorskip("langfuse")
CallbackHandler = pytest.importorskip("langfuse.langchain").CallbackHandler
OTLPSpanExporter = pytest.importorskip("opentelemetry.exporter.otlp.proto.http.trace_exporter").OTLPSpanExporter
_MEMORY = "Run database migrations before deploying the application."


class FixtureFailure(RuntimeError):
    """A deterministic model business failure, without external provider data."""


class RecordingCallback(CallbackHandler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lifecycle_events = []

    def _record_error(self, event, error, run_id, parent_run_id):
        self.lifecycle_events.append({
            "event": event,
            "error_type": type(error).__name__,
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id),
            "trace_id": f"{trace.get_current_span().get_span_context().trace_id:032x}",
        })

    def on_llm_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        self._record_error("on_llm_error", error, run_id, parent_run_id)
        return super().on_llm_error(error, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

    def on_chain_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        self._record_error("on_chain_error", error, run_id, parent_run_id)
        return super().on_chain_error(error, run_id=run_id, parent_run_id=parent_run_id, **kwargs)


class LifecycleModel(BaseChatModel):
    case: str
    ledger: Any
    started: Any
    closed: Any

    @property
    def _llm_type(self):
        return "deterministic-lifecycle-fixture"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, *args, **kwargs):
        raise NotImplementedError

    def _body_start(self):
        event = {
            "case": self.case,
            "start": time.time_ns(),
            "span_id": f"{trace.get_current_span().get_span_context().span_id:016x}",
        }
        self.ledger.append(event)
        self.started.set()
        return event

    async def _response(self, messages):
        if "cancel" in self.case:
            await asyncio.Future()
        if self.case == "exception":
            raise FixtureFailure()
        await asyncio.sleep(0.004)
        if self.case == "tool-loop" and not any(message.type == "tool" for message in messages):
            message = AIMessage(content="", tool_calls=[{"id": "fixture-tool", "name": "inspect_plan", "args": {}}])
        else:
            message = AIMessage(content="Verified deterministic deployment plan.")
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        event = self._body_start()
        try:
            result = await self._response(messages)
        except BaseException as error:
            event.update(error=type(error).__name__, end=time.time_ns())
            raise
        else:
            event["end"] = time.time_ns()
            return result
        finally:
            self.closed.set()

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        event = self._body_start()
        try:
            yield ChatGenerationChunk(message=AIMessageChunk(content="Partial local fixture output"))
            await asyncio.Future()
        except BaseException as error:
            event.update(error=type(error).__name__, end=time.time_ns())
            raise
        finally:
            self.closed.set()


@tool
async def inspect_plan() -> str:
    """Inspect a deterministic local deployment-plan fixture."""
    await asyncio.sleep(0.004)
    return "The local fixture plan was checked."


def _save_evidence(directory, spans, cases, callback):
    if not directory:
        return
    from google.protobuf.json_format import MessageToDict
    from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans

    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    encoded = encode_spans(spans)
    (output / "lifecycle.otlp.pb").write_bytes(encoded.SerializeToString())
    payload = MessageToDict(encoded)

    def ids_to_hex(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"traceId", "spanId", "parentSpanId"} and isinstance(item, str):
                    value[key] = base64.b64decode(item).hex()
                else:
                    ids_to_hex(item)
        elif isinstance(value, list):
            for item in value:
                ids_to_hex(item)

    ids_to_hex(payload)
    (output / "lifecycle.otlp.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    # Diagnostic inspection only: never change the consumer's private run maps.
    residual = [
        {"name": observation._otel_span.name, "recording": observation._otel_span.is_recording()}
        for observation in callback._runs.values()
    ]
    (output / "lifecycle.scenarios.json").write_text(
        json.dumps(
            {
                "evidence": "actual PowerContext ASGI/SQLite + official create_agent; deterministic async model; "
                "public inline callback plus per-invocation OTel boundary",
                "cases": cases,
                "dependencies": {
                    name: version(name) for name in ("powercontext", "langchain", "langchain-core", "langfuse")
                },
                "consumer_sources": {
                    name: {
                        "path": inspect.getfile(consumer),
                        "sha256": hashlib.sha256(Path(inspect.getfile(consumer)).read_bytes()).hexdigest(),
                    }
                    for name, consumer in (("chat_model", BaseChatModel), ("callback", CallbackHandler))
                },
                "callback_residuals": residual,
                "callback_context_token_count": len(callback._context_tokens),
                "callback_events": callback.lifecycle_events,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _assert_case(spans, case):
    own = sorted(
        [span for span in spans if f"{span.context.trace_id:032x}" == case["trace_id"]], key=lambda s: s.start_time
    )
    indexed = {span.context.span_id: span for span in own}
    generations = [span for span in own if span.name == "LifecycleModel"]
    recalls = [span for span in own if span.name == "PowerContextMiddleware recall"]
    captures = [span for span in own if span.name == "PowerContextClient capture_content_source"]
    events = case["model_bodies"]
    assert len(generations) == len(events), f"{case['case']}: actual started generation missing from export"
    assert len(recalls) == len(events)
    for generation, recall, body in zip(generations, recalls, events, strict=True):
        assert generation.end_time is not None and generation.end_time >= body["end"]
        assert generation.start_time <= body["start"]
        assert generation.parent is not None and recall.parent is not None
        assert generation.parent.span_id == recall.parent.span_id
        model_node = indexed[generation.parent.span_id]
        assert model_node.name == "model"
        assert model_node.start_time <= recall.start_time
        assert generation.end_time <= model_node.end_time
        assert recall.end_time <= generation.start_time
        assert f"{generation.context.span_id:016x}" == body["span_id"]
    if case["case"] in {"normal", "tool-loop", "parallel-a", "parallel-b"}:
        assert len(captures) == 1
        assert captures[0].start_time >= generations[-1].end_time
        capture_node = indexed[captures[0].parent.span_id]
        assert capture_node.name == "PowerContextMiddleware.after_agent"
        assert capture_node.start_time <= captures[0].start_time <= captures[0].end_time <= capture_node.end_time
    else:
        assert not captures, "an incomplete turn must not be captured"
        for generation in generations:
            assert generation.status.status_code.name == "ERROR"
            attrs = generation.attributes or {}
            assert attrs.get("langfuse.observation.level") == "ERROR"
            assert not attrs.get("langfuse.observation.output"), "do not invent a completed model response"
            if "cancel" in case["case"] or case["case"] == "stream-close":
                assert (
                    "CancelledError" in str(attrs.get("langfuse.observation.status_message", ""))
                    or attrs.get("error.type") == "CancelledError"
                )
                assert not any("usage_details" in key or "cost_details" in key for key in attrs)
    assert case["caller_restored"] and case["marker_cleared"]
    assert case["model_body_closed"]
    assert all(span.parent is None or span.parent.span_id in indexed for span in own)


async def _exercise_cases(invoke):
    for name in ["normal", "exception", "tool-loop", "cancel", "stream-cancel", "stream-close"]:
        started = asyncio.Event()
        task = asyncio.create_task(invoke(name, started))
        if "cancel" in name:
            await asyncio.wait_for(started.wait(), 2)
            task.cancel()
        result = (await asyncio.gather(task, return_exceptions=True))[0]
        if "cancel" in name:
            assert isinstance(result, asyncio.CancelledError)
        elif name == "exception":
            assert isinstance(result, RuntimeError)
        else:
            assert result is None, result
    started = asyncio.Event()
    parallel = [
        asyncio.create_task(invoke(name, started if name == "parallel-cancel" else None))
        for name in ["parallel-a", "parallel-b", "parallel-cancel"]
    ]
    await asyncio.wait_for(started.wait(), 2)
    parallel[-1].cancel()
    results = await asyncio.gather(*parallel, return_exceptions=True)
    assert results[:2] == [None, None]
    assert isinstance(results[-1], asyncio.CancelledError)


def test_actual_agent_generation_lifecycle_exports_completed_error_and_cancelled_runs():
    sink = InMemorySpanExporter()
    resource = {
        "review.evidence": "actual-asgi-sqlite-lifecycle-fixture",
        "ant.agent.product": "local-review-cancellation",
    }
    provider = TracerProvider(
        resource=Resource.create({**resource, "service.name": "PowerContext lifecycle application"}),
        shutdown_on_exit=False,
    )
    server_provider = TracerProvider(
        resource=Resource.create({**resource, "service.name": "PowerContext lifecycle server"}), shutdown_on_exit=False
    )
    server_provider.add_span_processor(SimpleSpanProcessor(sink))
    cases = []
    client = None
    key = "local-lifecycle-" + uuid4().hex
    with (
        tempfile.TemporaryDirectory() as directory,
        patch.dict(
            os.environ,
            {
                "LANGFUSE_PUBLIC_KEY": key,
                "LANGFUSE_SECRET_KEY": "local-test-only",
                "LANGFUSE_BASE_URL": "http://127.0.0.1:9",
                "LANGFUSE_MEDIA_UPLOAD_ENABLED": "false",
                "LANGSMITH_TRACING": "false",
                "OPENAI_API_KEY": "local-test-only",
                "OPENAI_MODEL": "local-fixture",
                "OPENAI_BASE_URL": "http://127.0.0.1:9",
                "POWERCONTEXT_LANGCHAIN_BASE_URL": "http://testserver",
                "POWERCONTEXT_LANGCHAIN_ALLOW_INSECURE_HTTP": "true",
            },
        ),
        patch("requests.sessions.Session.send", side_effect=AssertionError("external network forbidden")),
        patch.object(OTLPSpanExporter, "export", side_effect=sink.export),
        patch.object(trace, "get_tracer", side_effect=provider.get_tracer),
    ):
        app = create_server_app(
            settings=ServerSettings(
                database=SQLiteConfig(url=f"sqlite+aiosqlite:///{directory}/runtime.db"),
                inference=InferenceConfig(generation_model="test"),
                runtime=RuntimeConfig(artifact_processing_families=()),
                mcp=McpConfig(enabled=False),
            ),
            tracing=ServerTracing(server_provider),
        )

        async def run():
            nonlocal client
            async with (
                app.router.lifespan_context(app),
                httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as http,
            ):
                with shared_http_client(http, trust_transport_security=True):
                    sdk = PowerContextClient("http://testserver", http_client=http, trust_transport_security=True)
                    scope = await sdk.resolve_scope_binding(ResolveScopeBindingRequest())
                    await sdk.remember_memory(
                        RememberMemoryRequest(
                            scope_id=scope.scope_id,
                            kind="decision",
                            text=_MEMORY,
                            description="Explicit local fixture memory",
                        )
                    )
                    client = langfuse.Langfuse(tracer_provider=provider)
                    callback = RecordingCallback(public_key=key)
                    callback.run_inline = True
                    middleware = PowerContextMiddleware(auto_capture=True, trace_content=True)

                    async def invoke(name, started=None):
                        ledger = []
                        started, closed = started or asyncio.Event(), asyncio.Event()
                        model = LifecycleModel(case=name, ledger=ledger, started=started, closed=closed)
                        agent = create_agent(
                            model,
                            tools=[inspect_plan] if name == "tool-loop" else [],
                            middleware=[middleware],
                            context_schema=PowerContextScope,
                        )
                        record = {"case": name, "model_bodies": ledger}
                        with client.start_as_current_observation(
                            name="Actual lifecycle: " + name, as_type="agent"
                        ) as root:
                            record["trace_id"] = root.trace_id
                            root_span = trace.get_current_span().get_span_context()
                            try:
                                with trace.use_span(
                                    trace.get_current_span(),
                                    end_on_exit=False,
                                    record_exception=False,
                                    set_status_on_exception=False,
                                ):
                                    if name.startswith("stream-"):
                                        stream = agent.astream(
                                            {"messages": [{"role": "user", "content": _MEMORY}]},
                                            config={"callbacks": [callback]},
                                            context=PowerContextScope(scope_id=scope.scope_id),
                                            stream_mode="messages",
                                        )
                                        try:
                                            await anext(stream)
                                            if name == "stream-close":
                                                await stream.aclose()
                                            else:
                                                async for _ in stream:
                                                    pass
                                        finally:
                                            await stream.aclose()
                                    else:
                                        await agent.ainvoke(
                                            {"messages": [{"role": "user", "content": _MEMORY}]},
                                            config={"callbacks": [callback]},
                                            context=PowerContextScope(scope_id=scope.scope_id),
                                        )
                            finally:
                                record["caller_restored"] = trace.get_current_span().get_span_context() == root_span
                                record["marker_cleared"] = context.get_value("powercontext.genai_client") is None
                                record["model_body_closed"] = closed.is_set()
                                record["consumer_early_close"] = name == "stream-close"
                                cases.append(record)
                        assert not trace.get_current_span().get_span_context().is_valid
                        await asyncio.wait_for(closed.wait(), 2)

                    await _exercise_cases(invoke)
                    client.flush()
                    selected = {case["trace_id"] for case in cases}
                    spans = [span for span in sink.get_finished_spans() if f"{span.context.trace_id:032x}" in selected]
                    _save_evidence(os.getenv("POWERCONTEXT_LIFECYCLE_EVIDENCE_DIR"), spans, cases, callback)
                    for case in cases:
                        _assert_case(spans, case)
                    assert not callback._runs, "a completed invocation must not retain active observations"
                    assert not callback._context_tokens

        try:
            asyncio.run(run())
        finally:
            if client is not None:
                client.shutdown()
            provider.shutdown()
            server_provider.shutdown()
