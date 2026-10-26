"""Loaders: raw files -> markdown text.

Markdown is the interchange format, so .md, .txt and .docx all go through
the same splitter. For Word files this module recovers structure from, in
order of reliability:

  1. real Heading / Title styles and outline levels (resolved through style
     inheritance, so styled headings are found even without direct formatting)
  2. real list styles and nesting levels
  3. visual cues, measured RELATIVE to the document's own body text:
       - noticeably larger font  -> heading (level by size rank)
       - indentation / typed markers -> bullet points, nested by indent level
       - short italic paragraph next to a table -> that table's caption
       - bold-only standalone line -> weak heading, only when the document
         has no other headings and the line introduces points or a table
  4. tables, emitted in document order as markdown tables

Inline bold is kept as **term**; other inline italics are dropped.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".docx"}

# Tunables for visual structure detection
HEADING_SIZE_RATIO = 1.15  # heading font >= body * ratio ...
HEADING_SIZE_MIN_DELTA = 1.5  # ... or >= body + this many points
HEADING_MAX_WORDS = 14
WEAK_HEADING_MAX_WORDS = 8
CAPTION_MAX_WORDS = 40
INDENT_TOLERANCE_EMU = 110_000  # ~0.12 inch: indents closer than this are the same level
TAB_EMU = 457_200  # 0.5 inch
SPACE_EMU = TAB_EMU // 4
MAX_POINT_DEPTH = 5


# --------------------------------------------------------------------------
# Plain text / markdown
# --------------------------------------------------------------------------

def load_text(file_path: Path) -> str:
    """Read .md/.txt robustly: BOM-safe, UTF-16 aware, cp1252/latin-1 fallback."""
    raw = Path(file_path).read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = raw.decode("utf-16")
    else:
        for encoding in ("utf-8-sig", "cp1252"):
            try:
                text = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            text = raw.decode("latin-1")
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


# --------------------------------------------------------------------------
# DOCX: paragraph feature extraction
# --------------------------------------------------------------------------

_MARKER_RE = re.compile(r"^(?:(?P<sym>[•●◦▪▫■□‣⁃➢➤➜→·–—*-])|(?P<num>\d{1,3})[.)]|(?P<let>[a-z])\))\s+")
_LIST_STYLE_RE = re.compile(r"list (?:bullet|number)(?: (\d))?")


@dataclass
class _Para:
    text: str  # clean plain text
    md: str  # text with **bold** markers
    size: float  # dominant font size in points
    bold_frac: float
    italic_frac: float
    indent: int  # effective left indent in EMU (incl. leading tabs/spaces)
    heading_level: int = 0  # >0 for real styled headings
    is_title: bool = False
    list_depth: Optional[int] = None  # real list items
    marker: Optional[str] = None  # "-" or "<n>." for typed markers
    # decided by role assignment
    role: str = "para"  # title | heading | point | para | caption
    level: int = 0  # heading level
    depth: int = 0  # point nesting depth


@dataclass
class _Table:
    lines: List[str]
    caption: Optional[str] = None


_Block = Union[_Para, _Table]


def _chain(style) -> Iterator:
    seen = 0
    while style is not None and seen < 20:
        yield style
        style = style.base_style
        seen += 1


def _run_flag(run, para_style, attr: str) -> bool:
    value = getattr(run, attr)
    if value is not None:
        return bool(value)
    for style in _chain(run.style):
        value = getattr(style.font, attr)
        if value is not None:
            return bool(value)
    for style in _chain(para_style):
        value = getattr(style.font, attr)
        if value is not None:
            return bool(value)
    return False


def _run_size(run, para_style, default: float) -> float:
    if run.font.size:
        return run.font.size.pt
    for style in _chain(run.style):
        if style.font.size:
            return style.font.size.pt
    for style in _chain(para_style):
        if style.font.size:
            return style.font.size.pt
    return default


def _default_size(doc) -> float:
    try:
        vals = doc.styles.element.xpath("w:docDefaults/w:rPrDefault/w:rPr/w:sz/@w:val")
        if vals:
            return float(vals[0]) / 2
    except Exception:  # noqa: BLE001 - purely a fallback
        pass
    return 11.0


def _left_indent(para) -> int:
    value = para.paragraph_format.left_indent
    if value is not None:
        return int(value)
    for style in _chain(para.style):
        value = style.paragraph_format.left_indent
        if value is not None:
            return int(value)
    return 0


def _heading_level(para) -> Tuple[int, bool]:
    """(level, is_title) from the paragraph's style or outline level."""
    style = para.style
    name = (style.name or "").lower() if style is not None else ""
    sid = (style.style_id or "").lower() if style is not None else ""
    if name == "title" or sid == "title":
        return 0, True
    m = re.match(r"heading\s*(\d)", name) or re.match(r"heading(\d)", sid)
    if m:
        return min(int(m.group(1)), 6), False
    vals = para._p.xpath("./w:pPr/w:outlineLvl/@w:val")
    if not vals:
        for st in _chain(style):
            vals = st.element.xpath("./w:pPr/w:outlineLvl/@w:val")
            if vals:
                break
    if vals and str(vals[0]).isdigit() and int(vals[0]) <= 5:
        return int(vals[0]) + 1, False
    return 0, False


def _list_depth(para) -> Optional[int]:
    """Nesting depth for real (Word-tool) list items, else None."""
    num_ids = para._p.xpath("./w:pPr/w:numPr/w:numId/@w:val")
    if num_ids:
        if str(num_ids[0]) == "0":
            return None
        ilvl = para._p.xpath("./w:pPr/w:numPr/w:ilvl/@w:val")
        return int(ilvl[0]) if ilvl and str(ilvl[0]).isdigit() else 0
    style = para.style
    name = (style.name or "").lower() if style is not None else ""
    m = _LIST_STYLE_RE.match(name)
    if m:
        return int(m.group(1)) - 1 if m.group(1) else 0
    for st in _chain(style):
        if st.element.xpath("./w:pPr/w:numPr"):
            return 0
    return None


def _inline_md(para, whole_bold: bool) -> str:
    """Paragraph text with bold runs wrapped in ** ** (unless the whole line is bold)."""
    style = para.style
    parts: List[Tuple[bool, str]] = []
    for run in para.runs:
        if not run.text:
            continue
        bold = _run_flag(run, style, "bold")
        if parts and parts[-1][0] == bold:
            parts[-1] = (bold, parts[-1][1] + run.text)
        else:
            parts.append((bold, run.text))
    joined = "".join(t for _, t in parts)
    if " ".join(joined.split()) != " ".join(para.text.split()):
        return para.text  # runs miss content (e.g. hyperlinks): fall back to plain text
    out: List[str] = []
    for bold, text in parts:
        if bold and not whole_bold and text.strip():
            lead = text[: len(text) - len(text.lstrip())]
            trail = text[len(text.rstrip()):]
            out.append(f"{lead}**{text.strip()}**{trail}")
        else:
            out.append(text)
    return "".join(out)


def _analyze_paragraph(para, default_size: float) -> Optional[_Para]:
    raw = para.text
    text = " ".join(raw.split())
    if not text:
        return None

    style = para.style
    size_weight: Counter = Counter()
    bold_chars = italic_chars = total = 0
    for run in para.runs:
        n = len(run.text.strip())
        if not n:
            continue
        total += n
        size_weight[round(_run_size(run, style, default_size) * 2) / 2] += n
        if _run_flag(run, style, "bold"):
            bold_chars += n
        if _run_flag(run, style, "italic"):
            italic_chars += n
    size = size_weight.most_common(1)[0][0] if size_weight else default_size
    bold_frac = bold_chars / total if total else 0.0
    italic_frac = italic_chars / total if total else 0.0

    lead = raw[: len(raw) - len(raw.lstrip(" \t"))]
    indent = _left_indent(para) + lead.count("\t") * TAB_EMU + lead.count(" ") * SPACE_EMU

    level, is_title = _heading_level(para)
    list_depth = _list_depth(para)

    marker = None
    body = text
    m = _MARKER_RE.match(text)
    if m and list_depth is None and not level and not is_title:
        body = text[m.end():].strip()
        marker = f"{m.group('num')}." if m.group("num") else "-"

    md = _inline_md(para, whole_bold=bold_frac >= 0.8)
    md = re.sub(r"\s*\n\s*", " ", md).replace("\t", " ")
    md = " ".join(md.split())
    if marker is not None:
        md_m = _MARKER_RE.match(md)
        md = md[md_m.end():].strip() if md_m else body

    return _Para(
        text=body, md=md, size=size, bold_frac=bold_frac, italic_frac=italic_frac,
        indent=indent, heading_level=level, is_title=is_title, list_depth=list_depth, marker=marker,
    )


def _table_to_lines(table) -> List[str]:
    rows: List[List[str]] = []
    for row in table.rows:
        cells: List[str] = []
        previous = None
        for cell in row.cells:
            if previous is not None and cell._tc is previous:
                continue  # horizontally merged cell
            previous = cell._tc
            text = "; ".join(p.text.strip() for p in cell.paragraphs if p.text.strip())
            cells.append(" ".join(text.split()).replace("|", "\\|"))
        if any(cells):
            rows.append(cells)
    if not rows:
        return []
    ncols = max(len(r) for r in rows)
    rows = [r + [""] * (ncols - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * ncols) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return lines


# --------------------------------------------------------------------------
# DOCX: role assignment
# --------------------------------------------------------------------------

def _word_count(text: str) -> int:
    return len(text.split())


def _looks_like_heading_text(p: _Para) -> bool:
    return (
        _word_count(p.text) <= HEADING_MAX_WORDS
        and not p.text.rstrip().endswith((".", "!", "?", ";", ","))
    )


def _assign_headings(paras: List[_Para], body_size: float) -> None:
    styled = [p for p in paras if p.heading_level and _word_count(p.text) <= 25]
    for p in paras:
        if p.is_title:
            p.role = "title"
    for p in styled:
        p.role, p.level = "heading", p.heading_level

    # visual candidates: noticeably larger than body, short, not sentence-like
    candidates = [
        p for p in paras
        if p.role == "para" and not p.heading_level and p.list_depth is None and p.marker is None
        and (p.size >= body_size * HEADING_SIZE_RATIO or p.size >= body_size + HEADING_SIZE_MIN_DELTA)
        and _looks_like_heading_text(p)
    ]
    if not candidates:
        return

    if styled:
        by_level: dict = {}
        for p in styled:
            by_level.setdefault(p.level, []).append(p.size)
        level_size = {lvl: sorted(s)[len(s) // 2] for lvl, s in by_level.items()}
        deepest = max(level_size)
        for p in candidates:
            nearest = min(level_size, key=lambda lvl: abs(level_size[lvl] - p.size))
            if abs(level_size[nearest] - p.size) <= 1.5:
                level = nearest
            elif p.size > max(level_size.values()):
                level = 1
            else:
                level = min(deepest + 1, 6)
            p.role, p.level = "heading", level
    else:
        sizes = sorted({p.size for p in candidates}, reverse=True)
        for p in candidates:
            p.role, p.level = "heading", min(sizes.index(p.size) + 1, 4)


def _attach_captions(blocks: List[_Block]) -> None:
    """A short, mostly-italic paragraph directly next to a table is its caption."""
    for i, block in enumerate(blocks):
        if not isinstance(block, _Para) or block.role != "para":
            continue
        if block.italic_frac < 0.8 or _word_count(block.text) > CAPTION_MAX_WORDS:
            continue
        nxt = blocks[i + 1] if i + 1 < len(blocks) else None
        prv = blocks[i - 1] if i > 0 else None
        if isinstance(nxt, _Table) and nxt.caption is None:
            nxt.caption = block.text
            block.role = "caption"
        elif isinstance(prv, _Table) and prv.caption is None:
            prv.caption = block.text
            block.role = "caption"


def _assign_points(blocks: List[_Block]) -> None:
    paras = [b for b in blocks if isinstance(b, _Para)]
    undecided = [p for p in paras if p.role == "para"]

    # cluster effective indents -> levels (level 0 = smallest indent)
    indents = sorted({p.indent for p in undecided if p.list_depth is None})
    clusters: List[int] = []
    for value in indents:
        if not clusters or value - clusters[-1] > INDENT_TOLERANCE_EMU:
            clusters.append(value)

    def level_of(indent: int) -> int:
        best = 0
        for i, start in enumerate(clusters):
            if indent >= start - INDENT_TOLERANCE_EMU:
                best = i
        return best

    for p in undecided:
        if p.list_depth is not None:
            p.role, p.depth = "point", min(p.list_depth, MAX_POINT_DEPTH)
            continue
        level = level_of(p.indent)
        if p.marker is not None or level >= 1:
            p.role = "point"
            p.depth = level  # adjusted below
        # else: stays a normal paragraph

    manual = [p for p in undecided if p.role == "point" and p.list_depth is None]
    point_base = 0 if any(p.marker is not None and p.depth == 0 for p in manual) else 1
    for p in manual:
        p.depth = max(0, min(p.depth - point_base, MAX_POINT_DEPTH))


def _assign_weak_headings(blocks: List[_Block]) -> None:
    """Fallback for flat documents: a bold-only line introducing points or a table."""
    if any(isinstance(b, _Para) and b.role in ("heading", "title") for b in blocks):
        return
    for i, block in enumerate(blocks):
        if not isinstance(block, _Para) or block.role != "para":
            continue
        if block.bold_frac < 0.8 or _word_count(block.text) > WEAK_HEADING_MAX_WORDS:
            continue
        if block.text.rstrip().endswith((".", "!", "?")):
            continue
        nxt = blocks[i + 1] if i + 1 < len(blocks) else None
        if isinstance(nxt, _Table) or (isinstance(nxt, _Para) and nxt.role == "point"):
            block.role, block.level = "heading", 2


def _render(blocks: List[_Block]) -> str:
    out: List[str] = []
    point_run: List[str] = []

    def flush_points() -> None:
        if point_run:
            out.append("\n".join(point_run))
            point_run.clear()

    for block in blocks:
        if isinstance(block, _Table):
            flush_points()
            head = [f"Table: {block.caption}"] if block.caption else []
            out.append("\n".join(head + block.lines))
            continue
        if block.role == "caption":
            continue
        if block.role == "point":
            bullet = f"{block.marker}" if block.marker and block.marker != "-" else "-"
            point_run.append("  " * block.depth + f"{bullet} {block.md}")
            continue
        flush_points()
        if block.role == "title":
            out.append(f"# {block.text}")
        elif block.role == "heading":
            out.append(f"{'#' * max(1, min(block.level, 6))} {block.text}")
        else:
            out.append(block.md)
    flush_points()
    return "\n\n".join(out)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def load_docx_as_markdown(file_path: Path) -> str:
    """Convert a Word file to structured markdown (see module docstring)."""
    import docx
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = docx.Document(str(file_path))
    default_size = _default_size(doc)

    blocks: List[_Block] = []
    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):
            analyzed = _analyze_paragraph(Paragraph(child, doc._body), default_size)
            if analyzed is not None:
                blocks.append(analyzed)
        elif child.tag == qn("w:tbl"):
            lines = _table_to_lines(Table(child, doc._body))
            if lines:
                blocks.append(_Table(lines=lines))

    paras = [b for b in blocks if isinstance(b, _Para)]
    if not paras and not blocks:
        return ""

    weights: Counter = Counter()
    for p in paras:
        if p.list_depth is None and not p.heading_level and not p.is_title:
            weights[p.size] += len(p.text)
    body_size = weights.most_common(1)[0][0] if weights else default_size

    _assign_headings(paras, body_size)
    _attach_captions(blocks)
    _assign_points(blocks)
    _assign_weak_headings(blocks)
    return _render(blocks)


def load_document_as_markdown(file_path: Path) -> str:
    suffix = Path(file_path).suffix.lower()
    if suffix == ".docx":
        return load_docx_as_markdown(file_path)
    if suffix in {".md", ".markdown", ".txt"}:
        return load_text(file_path)
    raise ValueError(f"Unsupported file type: {suffix}")