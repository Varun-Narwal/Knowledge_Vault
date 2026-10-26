"""Structure-aware chunking of markdown for LLM extraction.

Goals, in priority order:
  1. HARD CAP: no chunk may exceed the context budget (derived from the
     extractor's num_ctx / num_predict, so a smaller context on a small GPU
     automatically means smaller chunks).
  2. Coherence: a chunk stays inside one section; a bullet with its
     nested children, a table, and a code block are never split unless
     they alone exceed the budget.
  3. Context: every chunk carries its heading breadcrumb (section_path),
     tracked with a heading stack so levels can't go stale.
  4. No tiny chunks: heading-only or near-empty sections merge into a
     neighbour rather than costing an LLM call of their own.

Works with no headings at all: it then packs paragraphs / bullet groups.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$")
_LIST_RE = re.compile(r"^(\s*)(?:[-*+]|\d+[.)])\s+\S")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_TABLE_SEP_RE = re.compile(r"^\|[\s:|-]+\|?$")
_CAPTION_RE = re.compile(r"^Table:\s")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")


# --------------------------------------------------------------------------
# Config and output types
# --------------------------------------------------------------------------

@dataclass
class SplitterConfig:
    max_chars: int = 3000  # hard cap per chunk
    max_words: int = 400  # hard cap per chunk
    target_words: int = 300  # preferred size when a section has to be split
    min_body_words: int = 60  # chunks with less body text get merged into a neighbour

    @property
    def target_chars(self) -> int:
        return int(self.max_chars * 0.75)

    @classmethod
    def from_context(
        cls,
        num_ctx: int,
        num_predict: int,
        prompt_chars: int,
        *,
        chars_per_token: float = 3.0,  # conservative: tables and code tokenize badly
        safety: float = 0.85,
        overhead_tokens: int = 120,  # user-message wrapper + section-path line
        max_words: int = 400,
        target_words: int = 300,
        min_body_words: int = 60,
    ) -> "SplitterConfig":
        """Derive the chunk cap from what is left of the context window after
        the system prompt and the reserved output tokens."""
        prompt_tokens = int(prompt_chars / chars_per_token) + overhead_tokens
        available = num_ctx - prompt_tokens - num_predict
        if available < 200:
            raise ValueError(
                f"num_ctx={num_ctx} leaves only {available} tokens for the chunk after the prompt "
                f"(~{prompt_tokens}) and num_predict={num_predict}. Raise num_ctx or lower num_predict."
            )
        max_chars = max(600, int(available * safety * chars_per_token))
        return cls(
            max_chars=max_chars,
            max_words=min(max_words, max(80, max_chars // 5)),
            target_words=target_words,
            min_body_words=min_body_words,
        )


@dataclass
class Chunk:
    source_file: str
    index: int
    text: str
    section_path: Tuple[str, ...] = ()
    body_words: int = 0

    @property
    def context(self) -> str:
        """Breadcrumb handed to the extractor as metadata (not as content)."""
        return " > ".join((self.source_file, *self.section_path))

    @property
    def chars(self) -> int:
        return len(self.text)

    @property
    def words(self) -> int:
        return len(self.text.split())


# --------------------------------------------------------------------------
# Step 1: parse markdown into blocks
# --------------------------------------------------------------------------

@dataclass
class _Block:
    kind: str  # heading | para | list | table | code
    text: str
    level: int = 0
    title: str = ""
    units: List[str] = field(default_factory=list)  # list items / table rows / code lines
    header: str = ""  # table caption + header rows, repeated when a table is split


def _parse_blocks(markdown: str) -> List[_Block]:
    lines = markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks: List[_Block] = []
    i, n = 0, len(lines)

    while i < n:
        line = lines[i]
        if not line.strip():
            i += 1
            continue

        fence = _FENCE_RE.match(line)
        if fence:
            marker = fence.group(1)
            code = [line]
            i += 1
            while i < n:
                code.append(lines[i])
                closing = lines[i].strip().startswith(marker)
                i += 1
                if closing:
                    break
            blocks.append(_Block("code", "\n".join(code), units=code))
            continue

        heading = _HEADING_RE.match(line)
        if heading:
            title = heading.group(2).strip()
            blocks.append(
                _Block("heading", f"{heading.group(1)} {title}", level=len(heading.group(1)), title=title)
            )
            i += 1
            continue

        if line.lstrip().startswith("|"):
            table: List[str] = []
            while i < n and lines[i].lstrip().startswith("|"):
                table.append(lines[i].strip())
                i += 1
            has_sep = len(table) > 1 and bool(_TABLE_SEP_RE.match(table[1]))
            header = table[:2] if has_sep else []
            rows = table[2:] if has_sep else table
            if blocks and blocks[-1].kind == "para" and "\n" not in blocks[-1].text and _CAPTION_RE.match(blocks[-1].text):
                header = [blocks.pop().text] + header
            blocks.append(_Block("table", "\n".join(header + rows), units=rows, header="\n".join(header)))
            continue

        if _LIST_RE.match(line):
            items: List[str] = []
            current: List[str] = []
            base: Optional[int] = None
            while i < n and lines[i].strip():
                match = _LIST_RE.match(lines[i])
                indent = len(match.group(1).expandtabs(4)) if match else None
                if match and base is None:
                    base = indent
                if match and indent is not None and base is not None and indent <= base:
                    if current:
                        items.append("\n".join(current))
                    current = [lines[i].rstrip()]
                else:
                    current.append(lines[i].rstrip())
                i += 1
            if current:
                items.append("\n".join(current))
            blocks.append(_Block("list", "\n".join(items), units=items))
            continue

        para: List[str] = []
        while (
            i < n
            and lines[i].strip()
            and not _FENCE_RE.match(lines[i])
            and not _HEADING_RE.match(lines[i])
            and not lines[i].lstrip().startswith("|")
            and not _LIST_RE.match(lines[i])
        ):
            para.append(lines[i].rstrip())
            i += 1
        blocks.append(_Block("para", "\n".join(para)))

    return blocks


# --------------------------------------------------------------------------
# Step 2: break oversize blocks into atoms that respect the cap
# --------------------------------------------------------------------------

def _words(text: str) -> int:
    return len(text.split())


def _fits(text: str, cfg: SplitterConfig) -> bool:
    return len(text) <= cfg.max_chars and _words(text) <= cfg.max_words


def _hard_split(text: str, limit_chars: int, limit_words: int) -> List[str]:
    """Last resort: split on whitespace into pieces within BOTH the char and word limits."""
    pieces: List[str] = []
    current: List[str] = []
    current_len = 0
    for word in text.split():
        while len(word) > limit_chars:  # a single giant token (URL, hash, ...)
            if current:
                pieces.append(" ".join(current))
                current, current_len = [], 0
            pieces.append(word[:limit_chars])
            word = word[limit_chars:]
        added = len(word) + (1 if current else 0)
        if current and (current_len + added > limit_chars or len(current) + 1 > limit_words):
            pieces.append(" ".join(current))
            current, current_len = [word], len(word)
        else:
            current.append(word)
            current_len += added
    if current:
        pieces.append(" ".join(current))
    return pieces


def _pack(units: List[str], limit_chars: int, limit_words: int, joiner: str) -> List[str]:
    """Greedily pack units into pieces within the char and word limits (oversize units are hard-split)."""
    pieces: List[str] = []
    current = ""
    for unit in units:
        fits = len(unit) <= limit_chars and _words(unit) <= limit_words
        for part in [unit] if fits else _hard_split(unit, limit_chars, limit_words):
            joined = f"{current}{joiner}{part}" if current else part
            if current and (len(joined) > limit_chars or _words(joined) > limit_words):
                pieces.append(current)
                current = part
            else:
                current = joined
    if current:
        pieces.append(current)
    return pieces


def _atoms(block: _Block, cfg: SplitterConfig) -> List[str]:
    if block.kind == "heading" or _fits(block.text, cfg):
        return [block.text]
    chars, wrds = cfg.target_chars, cfg.target_words  # preferred size for pieces of an oversize block

    if block.kind == "para":
        sentences = [x for x in _SENTENCE_RE.split(block.text) if x.strip()]
        return _pack(sentences, chars, wrds, " ")

    if block.kind == "list":
        atoms: List[str] = []
        for item in block.units:
            atoms.extend([item] if _fits(item, cfg) else _pack(item.split("\n"), chars, wrds, "\n"))
        return atoms

    if block.kind == "table":
        head = block.header
        room_chars = max(200, chars - len(head) - 1)
        room_words = max(20, wrds - _words(head))
        return [f"{head}\n{piece}" if head else piece for piece in _pack(block.units, room_chars, room_words, "\n")]

    # code: re-fence each piece so it stays a valid block
    inner = [ln for ln in block.units if not _FENCE_RE.match(ln)]
    return [f"```\n{piece}\n```" for piece in _pack(inner, chars - 8, wrds, "\n")]


# --------------------------------------------------------------------------
# Step 3: sections -> drafts (pack atoms), then merge small drafts
# --------------------------------------------------------------------------

@dataclass
class _Draft:
    text: str
    path: Tuple[str, ...]
    body_words: int


def _section_drafts(
    path: Tuple[str, ...], heading_line: Optional[str], blocks: List[_Block], cfg: SplitterConfig
) -> List[_Draft]:
    atoms: List[Tuple[str, bool]] = []  # (text, is_heading)
    if heading_line:
        atoms.append((heading_line, True))
    for block in blocks:
        atoms.extend((text, False) for text in _atoms(block, cfg))
    if not atoms:
        return []

    def body(items: List[Tuple[str, bool]]) -> int:
        return sum(_words(t) for t, is_h in items if not is_h)

    total_chars = sum(len(t) for t, _ in atoms) + 2 * (len(atoms) - 1)
    total_words = sum(_words(t) for t, _ in atoms)
    if total_chars <= cfg.max_chars and total_words <= cfg.max_words:
        return [_Draft("\n\n".join(t for t, _ in atoms), path, body(atoms))]

    drafts: List[_Draft] = []
    current: List[Tuple[str, bool]] = []

    def size(items: List[Tuple[str, bool]]) -> Tuple[int, int]:
        return sum(len(t) for t, _ in items) + 2 * max(0, len(items) - 1), sum(_words(t) for t, _ in items)

    def flush() -> None:
        if current:
            drafts.append(_Draft("\n\n".join(t for t, _ in current), path, body(current)))
            current.clear()

    for atom in atoms:
        trial = current + [atom]
        chars, words = size(trial)
        over_target = chars > cfg.target_chars or words > cfg.target_words
        over_max = chars > cfg.max_chars or words > cfg.max_words
        heading_only = len(current) == 1 and current[0][1]
        if current and (over_max or (over_target and not heading_only)):
            flush()
        current.append(atom)
    flush()
    return drafts


def _related(a: Tuple[str, ...], b: Tuple[str, ...]) -> bool:
    if a[: len(b)] == b or b[: len(a)] == a:
        return True  # ancestor / descendant
    return a[:-1] == b[:-1]  # siblings


def _merged_path(a: Tuple[str, ...], b: Tuple[str, ...]) -> Tuple[str, ...]:
    if a[: len(b)] == b:
        return a
    if b[: len(a)] == a:
        return b
    common: List[str] = []
    for x, y in zip(a, b):
        if x != y:
            break
        common.append(x)
    return tuple(common)


def _can_merge(a: _Draft, b: _Draft, cfg: SplitterConfig) -> bool:
    text = f"{a.text}\n\n{b.text}"
    return _related(a.path, b.path) and _fits(text, cfg)


def _merge(a: _Draft, b: _Draft) -> _Draft:
    return _Draft(f"{a.text}\n\n{b.text}", _merged_path(a.path, b.path), a.body_words + b.body_words)


def _merge_small(drafts: List[_Draft], cfg: SplitterConfig) -> List[_Draft]:
    out: List[_Draft] = []
    i = 0
    while i < len(drafts):
        current = drafts[i]
        i += 1
        while current.body_words < cfg.min_body_words and i < len(drafts) and _can_merge(current, drafts[i], cfg):
            current = _merge(current, drafts[i])
            i += 1
        if current.body_words < cfg.min_body_words and out and _can_merge(out[-1], current, cfg):
            out[-1] = _merge(out[-1], current)
        else:
            out.append(current)
    return out


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def split_markdown(markdown: str, source_file: str, cfg: Optional[SplitterConfig] = None) -> List[Chunk]:
    """Split markdown into budget-safe chunks carrying their section breadcrumb."""
    cfg = cfg or SplitterConfig()
    blocks = _parse_blocks(markdown)

    drafts: List[_Draft] = []
    stack: List[Tuple[int, str]] = []
    path: Tuple[str, ...] = ()
    heading_line: Optional[str] = None
    body: List[_Block] = []

    def close_section() -> None:
        drafts.extend(_section_drafts(path, heading_line, body, cfg))

    for block in blocks:
        if block.kind == "heading":
            close_section()
            while stack and stack[-1][0] >= block.level:
                stack.pop()
            stack.append((block.level, block.title))
            path = tuple(title for _, title in stack)
            heading_line = block.text
            body = []
        else:
            body.append(block)
    close_section()

    chunks: List[Chunk] = []
    for draft in _merge_small(drafts, cfg):
        if draft.body_words == 0:
            continue  # heading-only leftovers carry nothing to extract
        chunks.append(
            Chunk(
                source_file=source_file,
                index=len(chunks),
                text=draft.text,
                section_path=draft.path,
                body_words=draft.body_words,
            )
        )
    return chunks