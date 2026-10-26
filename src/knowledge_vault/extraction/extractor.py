from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Tuple
from pathlib import Path
import hashlib, time

import httpx
from pydantic import BaseModel, Field, ValidationError

from knowledge_vault.extraction.prompts import EXTRACTION_SYSTEM_PROMPT
from knowledge_vault.extraction.schemas import (
    ExtractedEdge,
    ExtractedNode,
    ExtractionGraphPayload,
    build_llm_output_schema,
)

logger = logging.getLogger(__name__)

ErrorKind = Literal["transport", "fatal", "timeout", "truncated", "parse", "schema", "unexpected"]
MAX_REASONS = 20


# --------------------------------------------------------------------------
# Config and outcome types
# --------------------------------------------------------------------------

@dataclass
class ExtractorConfig:
    base_url: str = "http://localhost:11434"
    model: str = "qwen3:8b"
    num_ctx: int = 4096
    num_predict: int = 2048  # output cap; also what turns a runaway generation into a "length" signal
    temperature: float = 0.0
    top_p: float = 0.1
    retry_temperature: float = 0.3
    connect_timeout: float = 5.0  # fail fast when Ollama is down
    read_timeout: float = 180.0  # whole non-streamed generation must finish within this
    max_transport_retries: int = 2
    backoff_seconds: Tuple[float, ...] = (2.0, 5.0)
    max_split_depth: int = 2  # a chunk is split into at most 4 pieces
    min_words_to_split: int = 40
    think: bool = False
    debug_dir: Optional[Path] = None

    @classmethod
    def from_settings(cls, **overrides: Any) -> "ExtractorConfig":
        """Build from knowledge_vault.config (imported lazily so this module
        works without it, e.g. in tests)."""
        from knowledge_vault.config import EXTRACTION_MODEL, MAX_CONTEXT_TOKENS, OLLAMA_BASE_URL

        return cls(
            base_url=OLLAMA_BASE_URL,
            model=EXTRACTION_MODEL,
            num_ctx=MAX_CONTEXT_TOKENS,
            **overrides,
        )


class ExtractionOutcome(BaseModel):
    """Result of extracting one chunk. JSON-serializable so the driver can
    cache it and later re-run only the failed chunks."""

    status: Literal["ok", "partial", "failed"]
    payload: ExtractionGraphPayload = Field(default_factory=ExtractionGraphPayload)
    rejected_items: int = 0
    rejection_reasons: List[str] = Field(default_factory=list)  # capped sample
    attempts: int = 0  # total HTTP requests made for this chunk (including split pieces)
    pieces: int = 1  # >1 when the chunk had to be split
    error_kind: Optional[ErrorKind] = None
    error_message: Optional[str] = None

    @property
    def fatal(self) -> bool:
        """True for configuration-level failures the driver should abort on."""
        return self.status == "failed" and self.error_kind == "fatal"


def _failed(kind: ErrorKind, message: str, attempts: int) -> ExtractionOutcome:
    return ExtractionOutcome(status="failed", error_kind=kind, error_message=message, attempts=attempts)


# --------------------------------------------------------------------------
# Chunk splitting
# --------------------------------------------------------------------------

def split_text(text: str, min_words: int = 40) -> Optional[Tuple[str, str]]:
    """Split text into two halves at the boundary nearest the middle,
    preferring paragraph, then sentence, then word boundaries. Returns None
    if the text is too short to split."""
    if len(text.split()) < min_words:
        return None

    half_min = max(1, min_words // 2)
    middle = len(text) / 2

    for pattern in (r"\n\s*\n", r"(?<=[.!?])\s+", r"\s+"):
        cuts = [m.end() for m in re.finditer(pattern, text)]
        if not cuts:
            continue
        cut = min(cuts, key=lambda c: abs(c - middle))
        left, right = text[:cut].strip(), text[cut:].strip()
        if len(left.split()) >= half_min and len(right.split()) >= half_min:
            return left, right
    return None


# --------------------------------------------------------------------------
# Parsing and item-level validation
# --------------------------------------------------------------------------

class _ParseFailure(Exception):
    def __init__(self, kind: ErrorKind, message: str, retryable: bool):
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.retryable = retryable


def _summarize(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in e['loc']) or 'item'}: {e['msg']}" for e in exc.errors()
    )


def _validate_nodes(raw: Any, reasons: List[str]) -> Tuple[List[ExtractedNode], int]:
    if raw is None:
        return [], 0
    if not isinstance(raw, list):
        reasons.append("nodes: expected a list")
        return [], 1
    nodes: List[ExtractedNode] = []
    rejected = 0
    for i, item in enumerate(raw):
        if isinstance(item, dict) and item.get("entity_type") is None:
            item = {**item, "entity_type": "Other"}  # don't lose a node over a missing type
        try:
            node = ExtractedNode.model_validate(item)
        except ValidationError as exc:
            rejected += 1
            reasons.append(f"node[{i}]: {_summarize(exc)}")
            continue
        name = node.name.strip()
        if not name:
            rejected += 1
            reasons.append(f"node[{i}]: name is blank")
            continue
        nodes.append(node.model_copy(update={"name": name}))
    return nodes, rejected


def _validate_edges(raw: Any, reasons: List[str]) -> Tuple[List[ExtractedEdge], int]:
    if raw is None:
        return [], 0
    if not isinstance(raw, list):
        reasons.append("edges: expected a list")
        return [], 1
    edges: List[ExtractedEdge] = []
    rejected = 0
    for i, item in enumerate(raw):
        try:
            edge = ExtractedEdge.model_validate(item)
        except ValidationError as exc:
            rejected += 1
            reasons.append(f"edge[{i}]: {_summarize(exc)}")
            continue
        source, target, rel = edge.source.strip(), edge.target.strip(), edge.relationship_type.strip()
        blank = [n for n, v in (("source", source), ("target", target), ("relationship_type", rel)) if not v]
        if blank:
            rejected += 1
            reasons.append(f"edge[{i}]: blank {', '.join(blank)}")
            continue
        edges.append(edge.model_copy(update={"source": source, "target": target, "relationship_type": rel}))
    return edges, rejected


def _parse_and_validate(content: str) -> Tuple[ExtractionGraphPayload, int, List[str]]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise _ParseFailure("parse", f"invalid JSON: {exc.msg} (char {exc.pos})", retryable=True)
    if not isinstance(data, dict):
        raise _ParseFailure("schema", "top-level JSON is not an object", retryable=False)

    reasons: List[str] = []
    nodes, rejected_nodes = _validate_nodes(data.get("nodes"), reasons)
    edges, rejected_edges = _validate_edges(data.get("edges"), reasons)
    return (
        ExtractionGraphPayload(nodes=nodes, edges=edges),
        rejected_nodes + rejected_edges,
        reasons[:MAX_REASONS],
    )


# --------------------------------------------------------------------------
# The extractor
# --------------------------------------------------------------------------

@dataclass
class _Call:
    attempts: int
    content: str = ""
    thinking: str = ""
    done_reason: Optional[str] = None
    eval_count: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    total_seconds: Optional[float] = None
    error_kind: Optional[str] = None  # "transport" | "fatal" | "timeout"
    error_message: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error_kind is None


def _error_text(resp: httpx.Response) -> str:
    try:
        body = resp.json()
        if isinstance(body, dict) and body.get("error"):
            return str(body["error"])
    except ValueError:
        pass
    return resp.text[:200]


class OllamaExtractor:
    """Holds one shared HTTP client for a whole ingestion run.

    Use as an async context manager, or call `aclose()` when done. Pass
    `client=` to inject a client (tests use httpx.MockTransport).
    """

    def __init__(
        self,
        config: Optional[ExtractorConfig] = None,
        client: Optional[httpx.AsyncClient] = None,
    ):
        self.cfg = config or ExtractorConfig.from_settings()
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=self.cfg.connect_timeout,
                read=self.cfg.read_timeout,
                write=10.0,
                pool=10.0,
            )
        )
        self._endpoint = f"{self.cfg.base_url.rstrip('/')}/api/chat"
        self._schema = build_llm_output_schema()

    def _dump_truncation(self, text: str, call: _Call, temperature: float, depth: int, context: Optional[str]) -> None:
        logger.warning(
            "TRUNCATED depth=%d words=%d prompt_tok=%s out_tok=%s thinking_chars=%d content_chars=%d",
            depth, len(text.split()), call.prompt_eval_count, call.eval_count,
            len(call.thinking), len(call.content),
        )
        if not self.cfg.debug_dir:
            return
        self.cfg.debug_dir.mkdir(parents=True, exist_ok=True)
        tag = hashlib.sha1(text.encode()).hexdigest()[:10]
        record = {
            "context": context, "depth": depth, "temperature": temperature,
            "input_words": len(text.split()), "prompt_eval_count": call.prompt_eval_count,
            "eval_count": call.eval_count, "seconds": call.total_seconds,
            "thinking": call.thinking, "content": call.content, "input_text": text,
        }
        (self.cfg.debug_dir / f"{tag}_d{depth}_{int(time.time())}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        
    async def __aenter__(self) -> "OllamaExtractor":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # ---- public API ------------------------------------------------------

    async def extract(self, text: str, context: Optional[str] = None) -> ExtractionOutcome:
        """`context` is the chunk's section breadcrumb, sent as metadata with every request."""
        try:
            return await self._extract(text, depth=0, context=context)
        except Exception as exc:  # keep one surprising chunk from killing a long run
            logger.exception("Unexpected error while extracting a chunk")
            return _failed("unexpected", f"{type(exc).__name__}: {exc}", attempts=0)

    # ---- internals ---------------------------------------------------------

    def _request_body(self, text: str, temperature: float, context: Optional[str] = None) -> dict:
        meta = f"Section path: {context}\n\n" if context else ""
        return {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": f"{meta}Extract graph elements from this text:\n\n{text}"},
            ],
            "format": self._schema,
            "stream": False,  # we want the full payload at once
            "think": self.cfg.think,
            "options": {
                "temperature": temperature,
                "num_ctx": self.cfg.num_ctx,
                "num_predict": self.cfg.num_predict,
                "top_p": self.cfg.top_p,
            },
        }

    async def _call_ollama(self, text: str, temperature: float, context: Optional[str] = None) -> _Call:
        """One logical request, with transport-level retries only."""
        body = self._request_body(text, temperature, context)
        max_attempts = self.cfg.max_transport_retries + 1
        last_message = ""

        for attempt in range(1, max_attempts + 1):
            try:
                resp = await self._client.post(self._endpoint, json=body)
            except httpx.ReadTimeout:
                # Retrying a long generation would just time out again; the caller splits instead.
                return _Call(attempts=attempt, error_kind="timeout", error_message="read timeout waiting for the model")
            except httpx.TransportError as exc:
                last_message = f"{type(exc).__name__}: {exc}"
            else:
                if resp.status_code >= 500:
                    last_message = f"HTTP {resp.status_code}: {_error_text(resp)}"
                elif resp.status_code >= 400:
                    return _Call(
                        attempts=attempt,
                        error_kind="fatal",
                        error_message=f"HTTP {resp.status_code}: {_error_text(resp)}",
                    )
                else:
                    try:
                        data = resp.json()
                    except ValueError:
                        data = None
                    if isinstance(data, dict):
                        message = data.get("message") or {}
                        if not isinstance(message, dict):
                            message = {}
                        return _Call(
                            attempts=attempt,
                            content=(message.get("content") or "") if isinstance(message, dict) else "",
                            thinking=message.get("thinking") or "",
                            done_reason=data.get("done_reason"),
                            eval_count=data.get("eval_count"),
                            prompt_eval_count=data.get("prompt_eval_count"),
                            total_seconds=(data.get("total_duration") or 0) / 1e9,
                        )
                    last_message = "Ollama returned a non-JSON response body"

            if attempt < max_attempts:
                backoff = self.cfg.backoff_seconds
                delay = backoff[min(attempt - 1, len(backoff) - 1)] if backoff else 0.0
                await asyncio.sleep(delay)

        return _Call(attempts=max_attempts, error_kind="transport", error_message=last_message)

    async def _extract(self, text: str, depth: int, context: Optional[str] = None) -> ExtractionOutcome:
        attempts = 0
        failure: Optional[_ParseFailure] = None

        # second pass only happens after a retryable parse failure
        for temperature in (self.cfg.temperature, self.cfg.retry_temperature):
            call = await self._call_ollama(text, temperature, context)
            attempts += call.attempts

            if not call.ok:
                if call.error_kind == "timeout":
                    return await self._split_or_fail(text, depth, "timeout", call.error_message or "", attempts, context)
                return _failed(call.error_kind, call.error_message or "", attempts)  # type: ignore[arg-type]

            if call.done_reason == "length":
                self._dump_truncation(text, call, temperature, depth, context)
                return await self._split_or_fail(
                    text, depth, "truncated", "output hit the token limit (num_predict / num_ctx)", attempts, context
                )

            try:
                payload, rejected, reasons = _parse_and_validate(call.content)
            except _ParseFailure as pf:
                failure = pf
                if not pf.retryable:
                    break
                continue

            return ExtractionOutcome(
                status="partial" if rejected else "ok",
                payload=payload,
                rejected_items=rejected,
                rejection_reasons=reasons,
                attempts=attempts,
            )

        assert failure is not None
        return _failed(failure.kind, failure.message, attempts)

    async def _split_or_fail(
        self, text: str, depth: int, kind: ErrorKind, message: str, attempts: int, context: Optional[str] = None
    ) -> ExtractionOutcome:
        halves = split_text(text, self.cfg.min_words_to_split) if depth < self.cfg.max_split_depth else None
        if halves is None:
            return _failed(kind, f"{message}; chunk could not be split any further", attempts)

        logger.info("Chunk too heavy (%s); splitting in two (depth %d)", kind, depth + 1)
        parts = [await self._extract(half, depth + 1, context) for half in halves]
        return _merge(parts, attempts_before=attempts)


def _merge(parts: List[ExtractionOutcome], attempts_before: int) -> ExtractionOutcome:
    failed_parts = [p for p in parts if p.status == "failed"]
    first_error = next((p for p in parts if p.error_kind), None)

    if len(failed_parts) == len(parts):
        status = "failed"
    elif failed_parts or any(p.status == "partial" for p in parts):
        status = "partial"
    else:
        status = "ok"

    reasons: List[str] = []
    for p in parts:
        reasons.extend(p.rejection_reasons)

    return ExtractionOutcome(
        status=status,
        payload=ExtractionGraphPayload(
            nodes=[n for p in parts for n in p.payload.nodes],
            edges=[e for p in parts for e in p.payload.edges],
        ),
        rejected_items=sum(p.rejected_items for p in parts),
        rejection_reasons=reasons[:MAX_REASONS],
        attempts=attempts_before + sum(p.attempts for p in parts),
        pieces=sum(p.pieces for p in parts),
        error_kind=first_error.error_kind if first_error else None,
        error_message=first_error.error_message if first_error else None,
    )