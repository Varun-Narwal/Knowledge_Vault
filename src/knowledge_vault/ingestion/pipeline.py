"""Phase 1 pipeline: files -> chunks -> extraction -> resolution -> graph JSON.

    discover files -> markdown -> chunks            (no LLM, cheap)
    for each chunk: extract (or reuse cached outcome)
    resolve once over everything                    (pure function)
    write graph.json, resolver_report.json, run_summary.json, ...

Resilience
  * every extraction outcome is cached to disk the moment it is produced, so
    an interrupted or aborted run resumes where it stopped;
  * a chunk's cache key covers model + prompt + schema + text, so editing a
    note (or the prompt) only re-extracts what changed;
  * failed chunks are retried on the next run; ok/partial ones are reused;
  * one bad file or chunk never kills the run. Only a fatal server error or
    repeated "Ollama unreachable" failures abort it.

Everything written under out_dir is derived from personal notes: keep it
out of version control.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

from pydantic import BaseModel, Field

from knowledge_vault.extraction.extractor import ExtractionOutcome, ExtractorConfig, OllamaExtractor
from knowledge_vault.extraction.prompts import EXTRACTION_SYSTEM_PROMPT
from knowledge_vault.extraction.resolved_schemas import ChunkExtraction
from knowledge_vault.extraction.resolver import resolve
from knowledge_vault.extraction.schemas import build_llm_output_schema
from knowledge_vault.ingestion.loaders import SUPPORTED_SUFFIXES, load_document_as_markdown
from knowledge_vault.ingestion.logical_splitter import Chunk, SplitterConfig, split_markdown

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Files -> chunks
# --------------------------------------------------------------------------

@dataclass
class SkippedFile:
    path: str
    reason: str


def discover_files(raw_dir: Path) -> Tuple[List[Path], List[SkippedFile]]:
    """Sorted, reproducible file list. Hidden files/dirs are ignored silently;
    Word lock files and unsupported types are reported."""
    files: List[Path] = []
    skipped: List[SkippedFile] = []
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(raw_dir)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if path.name.startswith("~$"):
            skipped.append(SkippedFile(rel.as_posix(), "Word lock file"))
        elif path.suffix.lower() not in SUPPORTED_SUFFIXES:
            skipped.append(SkippedFile(rel.as_posix(), f"unsupported type {path.suffix or '(none)'}"))
        else:
            files.append(path)
    return files, skipped


def build_chunks(
    raw_dir: Path, splitter_cfg: SplitterConfig, markdown_dir: Optional[Path] = None
) -> Tuple[List[Chunk], List[SkippedFile]]:
    """Load and split every supported file. A file that fails to load is
    skipped and reported; it never stops the run. If `markdown_dir` is given,
    the converted markdown is saved there so you can inspect what structure
    was detected."""
    files, skipped = discover_files(raw_dir)
    chunks: List[Chunk] = []
    for path in files:
        rel = path.relative_to(raw_dir).as_posix()
        try:
            markdown = load_document_as_markdown(path)
            file_chunks = split_markdown(markdown, rel, splitter_cfg)
        except Exception as exc:  # noqa: BLE001 - isolate per-file failures
            logger.warning("Skipping %s: %s: %s", rel, type(exc).__name__, exc)
            skipped.append(SkippedFile(rel, f"load failed: {type(exc).__name__}: {exc}"))
            continue
        if markdown_dir is not None:
            target = markdown_dir / f"{rel}.md"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(markdown, encoding="utf-8")
        if not file_chunks:
            skipped.append(SkippedFile(rel, "no extractable text"))
        chunks.extend(file_chunks)
    return chunks, skipped


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------

def prompt_fingerprint(cfg: ExtractorConfig) -> str:
    material = (
        EXTRACTION_SYSTEM_PROMPT
        + json.dumps(build_llm_output_schema(), sort_keys=True)
        + json.dumps({"think": cfg.think, "num_predict": cfg.num_predict,
                      "temperature": cfg.temperature, "top_p": cfg.top_p}, sort_keys=True)
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def cache_key(chunk: Chunk, model: str, fingerprint: str) -> str:
    material = "\x00".join([model, fingerprint, chunk.context, chunk.text])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class ChunkCache:
    def __init__(self, root: Path):
        self.root = root

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> Optional[ExtractionOutcome]:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            return ExtractionOutcome.model_validate(record["outcome"])
        except Exception:  # noqa: BLE001 - a corrupt entry is just a miss
            logger.warning("Ignoring unreadable cache entry %s", path.name)
            return None

    def put(self, key: str, chunk: Chunk, outcome: ExtractionOutcome) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "source_file": chunk.source_file,
            "chunk_index": chunk.index,
            "context": chunk.context,
            "outcome": outcome.model_dump(mode="json"),
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)  # atomic: a crash never leaves a half-written entry


# --------------------------------------------------------------------------
# Run summary
# --------------------------------------------------------------------------

class RunSummary(BaseModel):
    raw_dir: str
    out_dir: str
    mode: str = "extract"  # extract | resolve-only | dry-run
    files: int = 0
    chunks_total: int = 0
    chunks_extracted: int = 0  # fresh LLM calls this run
    chunks_cached: int = 0
    chunks_partial: int = 0
    chunks_failed: int = 0
    chunks_missing: int = 0  # resolve-only: no cached result available
    chunks_split: int = 0  # needed splitting to fit
    rejected_items: int = 0
    nodes: int = 0
    edges: int = 0
    stub_nodes: int = 0
    failures: List[dict] = Field(default_factory=list)
    skipped_files: List[dict] = Field(default_factory=list)
    aborted: Optional[str] = None
    elapsed_seconds: float = 0.0
    config: Dict[str, object] = Field(default_factory=dict)


class _Abort(Exception):
    pass


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_manifest(path: Path, chunks: List[Chunk]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            row = {
                "source_file": c.source_file, "index": c.index, "context": c.context,
                "words": c.words, "chars": c.chars, "body_words": c.body_words,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------

async def run_pipeline(
    raw_dir: Path,
    out_dir: Path,
    extractor_cfg: ExtractorConfig,
    splitter_cfg: Optional[SplitterConfig] = None,
    aliases: Optional[Mapping[str, str]] = None,
    *,
    force: bool = False,
    resolve_only: bool = False,
    dry_run: bool = False,
    max_consecutive_transport_failures: int = 3,
    extractor: Optional[OllamaExtractor] = None,
) -> RunSummary:
    started = time.perf_counter()
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splitter_cfg = splitter_cfg or SplitterConfig.from_context(
        extractor_cfg.num_ctx, extractor_cfg.num_predict, len(EXTRACTION_SYSTEM_PROMPT)
    )
    mode = "dry-run" if dry_run else "resolve-only" if resolve_only else "extract"
    summary = RunSummary(
        raw_dir=str(raw_dir), out_dir=str(out_dir), mode=mode,
        config={"model": extractor_cfg.model, "num_ctx": extractor_cfg.num_ctx,
                "num_predict": extractor_cfg.num_predict, "splitter": asdict(splitter_cfg)},
    )

    # ---- chunking (no LLM) -----------------------------------------------------
    chunks, skipped = build_chunks(raw_dir, splitter_cfg, markdown_dir=out_dir / "markdown")
    summary.files = len({c.source_file for c in chunks})
    summary.chunks_total = len(chunks)
    summary.skipped_files = [asdict(s) for s in skipped]
    _write_manifest(out_dir / "chunks.jsonl", chunks)
    logger.info("%d chunk(s) from %d file(s); %d file(s) skipped", len(chunks), summary.files, len(skipped))

    if dry_run:
        summary.elapsed_seconds = round(time.perf_counter() - started, 2)
        _write_json(out_dir / "run_summary.json", summary.model_dump())
        return summary

    # ---- extraction (resumable) -------------------------------------------------
    cache = ChunkCache(out_dir / "cache")
    fingerprint = prompt_fingerprint(extractor_cfg)
    owns_extractor = extractor is None and not resolve_only
    if extractor is None and not resolve_only:
        extractor = OllamaExtractor(extractor_cfg)

    extractions: List[ChunkExtraction] = []
    consecutive_transport = 0
    try:
        for n, chunk in enumerate(chunks, 1):
            key = cache_key(chunk, extractor_cfg.model, fingerprint)
            outcome = None if force else cache.get(key)
            if outcome is not None and outcome.status == "failed":
                outcome = None  # failed chunks are retried
            source = "cached"

            if outcome is None:
                if resolve_only:
                    summary.chunks_missing += 1
                    logger.warning("[%d/%d] %s #%d: no cached result (resolve-only)", n, len(chunks), chunk.source_file, chunk.index)
                    continue
                assert extractor is not None
                t0 = time.perf_counter()
                outcome = await extractor.extract(chunk.text, context=chunk.context)
                cache.put(key, chunk, outcome)
                source = f"{time.perf_counter() - t0:.1f}s"
                summary.chunks_extracted += 1
            else:
                summary.chunks_cached += 1

            logger.info(
                "[%d/%d] %s #%d  %s  nodes=%d edges=%d (%s)",
                n, len(chunks), chunk.source_file, chunk.index, outcome.status,
                len(outcome.payload.nodes), len(outcome.payload.edges), source,
            )

            summary.rejected_items += outcome.rejected_items
            if outcome.pieces > 1:
                summary.chunks_split += 1
            if outcome.status == "partial":
                summary.chunks_partial += 1
            if outcome.status in ("partial", "failed") and outcome.error_kind:
                summary.failures.append({
                    "source_file": chunk.source_file, "chunk_index": chunk.index, "context": chunk.context,
                    "status": outcome.status, "error_kind": outcome.error_kind, "error_message": outcome.error_message,
                })
            if outcome.status == "failed":
                summary.chunks_failed += 1
                if outcome.fatal:
                    raise _Abort(f"Fatal server error: {outcome.error_message}")
                consecutive_transport = consecutive_transport + 1 if outcome.error_kind == "transport" else 0
                if consecutive_transport >= max_consecutive_transport_failures:
                    raise _Abort(
                        f"Ollama appears unreachable ({consecutive_transport} consecutive failures): {outcome.error_message}"
                    )
                continue
            consecutive_transport = 0
            extractions.append(
                ChunkExtraction(source_file=chunk.source_file, chunk_index=chunk.index, payload=outcome.payload)
            )
    except _Abort as abort:
        summary.aborted = str(abort)
        logger.error("Run aborted: %s", abort)
    finally:
        if owns_extractor and extractor is not None:
            await extractor.aclose()

    # ---- resolution --------------------------------------------------------------
    if summary.aborted is None:
        graph, report = resolve(extractions, aliases=aliases)
        _write_json(out_dir / "graph.json", json.loads(graph.model_dump_json()))
        _write_json(out_dir / "resolver_report.json", report.model_dump())
        summary.nodes, summary.edges, summary.stub_nodes = len(graph.nodes), len(graph.edges), len(report.stub_nodes)
        logger.info(
            "Resolved %d node(s), %d edge(s); %d merge(s), %d blocked pair(s), %d stub(s)",
            len(graph.nodes), len(graph.edges), len(report.merges), len(report.blocked_pairs), len(report.stub_nodes),
        )

    summary.elapsed_seconds = round(time.perf_counter() - started, 2)
    _write_json(out_dir / "run_summary.json", summary.model_dump())
    return summary