#!/usr/bin/env python3
"""Run Phase 1 over a folder of notes: load -> split -> extract -> resolve.

Examples
    python scripts/run_ingestion.py data/raw                 # full run (resumable)
    python scripts/run_ingestion.py data/raw --dry-run       # chunk only, no LLM: inspect out/markdown + out/chunks.jsonl
    python scripts/run_ingestion.py data/raw --resolve-only  # re-resolve from cached extractions (tune aliases/resolver fast)
    python scripts/run_ingestion.py data/raw --force         # ignore the cache, re-extract everything

Outputs (in --out, default ./out; derived from your notes, keep it gitignored)
    graph.json, resolver_report.json, run_summary.json, chunks.jsonl,
    markdown/ (what the loader detected), cache/ (per-chunk results)

Exit codes: 0 success, 1 finished but some chunks failed, 2 aborted.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from knowledge_vault.extraction.extractor import ExtractorConfig  # noqa: E402
from knowledge_vault.extraction.resolver import load_aliases  # noqa: E402
from knowledge_vault.ingestion.pipeline import run_pipeline  # noqa: E402

log = logging.getLogger("run_ingestion")


def build_extractor_config(args: argparse.Namespace) -> ExtractorConfig:
    try:
        cfg = ExtractorConfig.from_settings()
    except (ImportError, AttributeError) as exc:
        log.warning("Could not read knowledge_vault.config (%s); using built-in defaults", exc)
        cfg = ExtractorConfig()
    overrides = {
        "model": args.model,
        "base_url": args.ollama_url,
        "num_ctx": args.num_ctx,
        "num_predict": args.num_predict,
        "debug_dir": Path(args.out) / "debug",
    }
    return dataclasses.replace(cfg, **{k: v for k, v in overrides.items() if v is not None})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source_dir", type=Path, help="Folder of .docx / .md / .txt notes")
    parser.add_argument("--out", type=Path, default=Path("out"), help="Output folder (default: ./out)")
    parser.add_argument("--aliases", type=Path, help="Alias JSON (default: data/aliases.json if it exists)")
    parser.add_argument("--model", help="Override the extraction model")
    parser.add_argument("--ollama-url", help="Override the Ollama base URL")
    parser.add_argument("--num-ctx", type=int, help="Override the context window (chunk cap follows it)")
    parser.add_argument("--num-predict", type=int, help="Override the output token cap")
    parser.add_argument("--force", action="store_true", help="Ignore cached results and re-extract")
    parser.add_argument("--resolve-only", action="store_true", help="Skip the LLM; resolve from cached results")
    parser.add_argument("--dry-run", action="store_true", help="Load and chunk only; no LLM, no resolution")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if not args.source_dir.is_dir():
        log.error("Source folder not found: %s", args.source_dir)
        return 2

    alias_path = args.aliases or (Path("data/aliases.json") if Path("data/aliases.json").exists() else None)
    aliases = load_aliases(alias_path) if alias_path else None
    if alias_path:
        log.info("Using %d alias(es) from %s", len(aliases or {}), alias_path)

    try:
        summary = asyncio.run(
            run_pipeline(
                args.source_dir,
                args.out,
                build_extractor_config(args),
                aliases=aliases,
                force=args.force,
                resolve_only=args.resolve_only,
                dry_run=args.dry_run,
            )
        )
    except ValueError as exc:  # e.g. num_ctx too small for the prompt + output reservation
        log.error("%s", exc)
        return 2

    print("\n=== Run summary ===")
    print(f"mode: {summary.mode}   files: {summary.files}   chunks: {summary.chunks_total}")
    if summary.mode != "dry-run":
        print(f"extracted: {summary.chunks_extracted}   cached: {summary.chunks_cached}   "
              f"partial: {summary.chunks_partial}   failed: {summary.chunks_failed}   split: {summary.chunks_split}")
        if summary.chunks_missing:
            print(f"missing from cache (resolve-only): {summary.chunks_missing}")
        print(f"graph: {summary.nodes} nodes, {summary.edges} edges ({summary.stub_nodes} stub)")
    if summary.skipped_files:
        print(f"skipped files: {len(summary.skipped_files)} (see run_summary.json)")
    print(f"details: {args.out / 'run_summary.json'}")

    if summary.aborted:
        print(f"\nABORTED: {summary.aborted}\nProgress is cached; fix the problem and re-run to resume.")
        return 2
    return 1 if summary.chunks_failed else 0


if __name__ == "__main__":
    sys.exit(main())