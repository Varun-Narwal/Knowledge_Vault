import argparse, asyncio, copy, json, re, time
from collections import Counter
from pathlib import Path

import httpx

from knowledge_vault.extraction.extractor import ExtractorConfig
from knowledge_vault.extraction.prompts import EXTRACTION_SYSTEM_PROMPT
from knowledge_vault.extraction.schemas import build_llm_output_schema
from knowledge_vault.ingestion.logical_splitter import SplitterConfig
from knowledge_vault.ingestion.pipeline import build_chunks


def no_context_schema() -> dict:
    s = copy.deepcopy(build_llm_output_schema())
    s["properties"]["edges"]["items"]["properties"].pop("context", None)
    return s


VARIANTS = [
    # name, think, option overrides, schema factory
    ("A baseline (no think flag)", None, {}, build_llm_output_schema),
    ("B think=false", False, {}, build_llm_output_schema),
    ("C think=false,t=0.3,rp=1.1", False, {"temperature": 0.3, "top_p": 0.9, "repeat_penalty": 1.1}, build_llm_output_schema),
    ("D think=false,no edge ctx", False, {}, no_context_schema),
    ("E think=false,predict=6000", False, {"num_predict": 6000, "num_ctx": 8192}, build_llm_output_schema),
]


def analyse(content: str) -> str:
    names = re.findall(r'"name"\s*:\s*"([^"]*)"', content)
    srcs = re.findall(r'"source"\s*:\s*"([^"]*)"', content)
    dup = (1 - len(set(names)) / len(names)) if names else 0.0
    top = Counter(names).most_common(1)
    try:
        d = json.loads(content)
        valid = f"valid JSON: {len(d.get('nodes', []))} nodes / {len(d.get('edges', []))} edges"
    except Exception:
        valid = f"INVALID JSON (names seen={len(names)}, edge sources seen={len(srcs)})"
    return f"{valid}; dup-name ratio={dup:.0%}; most repeated={top}"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir")
    ap.add_argument("targets", nargs="+", help='"file.docx:INDEX", e.g. "THEORY OF SUPPLY.docx:3"')
    ap.add_argument("--only", help="variant letters to run, e.g. B or BD")
    a = ap.parse_args()

    cfg = ExtractorConfig.from_settings()
    splitter = SplitterConfig.from_context(cfg.num_ctx, cfg.num_predict, len(EXTRACTION_SYSTEM_PROMPT))
    chunks, _ = build_chunks(Path(a.raw_dir), splitter)
    by_id = {(c.source_file, c.index): c for c in chunks}

    async with httpx.AsyncClient(timeout=httpx.Timeout(600, connect=5)) as client:
        for target in a.targets:
            file, _, idx = target.rpartition(":")
            chunk = by_id.get((file, int(idx))) if idx.isdigit() else None
            if chunk is None:
                print(f"\n!! no chunk for {target!r}. Available: {sorted(by_id)[:8]} ...")
                continue
            print(f"\n######## {chunk.context} | #{chunk.index} | {chunk.words} words | {chunk.chars} chars")

            for name, think, overrides, schema_fn in VARIANTS:
                if a.only and name[0] not in a.only:
                    continue
                options = {"temperature": cfg.temperature, "top_p": cfg.top_p,
                           "num_ctx": cfg.num_ctx, "num_predict": cfg.num_predict, **overrides}
                body = {
                    "model": cfg.model, "stream": False, "format": schema_fn(), "options": options,
                    "messages": [
                        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                        {"role": "user", "content": f"Section path: {chunk.context}\n\nExtract graph elements from this text:\n\n{chunk.text}"},
                    ],
                }
                if think is not None:
                    body["think"] = think
                t0 = time.perf_counter()
                r = await client.post(f"{cfg.base_url.rstrip('/')}/api/chat", json=body)
                dt = time.perf_counter() - t0
                data = r.json()
                msg = data.get("message") or {}
                content, thinking = msg.get("content") or "", msg.get("thinking") or ""
                print(f"== {name}  ({dt:.0f}s)")
                print(f"   done_reason={data.get('done_reason')}  prompt_tok={data.get('prompt_eval_count')}  "
                      f"out_tok={data.get('eval_count')}  thinking_chars={len(thinking)}  content_chars={len(content)}")
                print(f"   {analyse(content)}")
                print(f"   tail: ...{content[-160:]!r}\n")


asyncio.run(main())