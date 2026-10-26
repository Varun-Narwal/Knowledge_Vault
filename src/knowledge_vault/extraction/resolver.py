"""Entity resolution for Knowledge Vault (Phase 1).

A pure function: pooled per-chunk extractions in, canonical graph plus an
audit report out. No state is carried between calls, so results are
deterministic and the resolver can be re-run cheaply over cached raw
extractions while you tune it.

Design principle: PRECISION OVER RECALL. A wrong merge silently corrupts
the graph; a missed merge leaves a visible duplicate. So we merge only
when sure, and report everything we declined to merge.

Matching tiers
    Tier 0  key normalization (+ optional user alias file)
    Tier 1  exact key match
    Tier 2  conservative fuzzy match (typos / spelling variants only),
            blocked by guards: substring, differing digits, short names.
            Blocked pairs are reported, never merged.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from rapidfuzz import fuzz

from knowledge_vault.extraction.schemas import ALLOWED_ENTITY_TYPES
from knowledge_vault.extraction.resolved_schemas import (
    BlockedPair,
    ChunkExtraction,
    MergeRecord,
    ResolutionReport,
    ResolvedEdge,
    ResolvedGraph,
    ResolvedNode,
    SourceRef,
)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

_TYPE_LOOKUP = {t.lower(): t for t in ALLOWED_ENTITY_TYPES}

DEFAULT_FUZZY_THRESHOLD = 92.0
DEFAULT_MIN_FUZZY_LEN = 5
DEFAULT_MAX_NAME_LEN = 80

_ARTICLE_PREFIX = re.compile(r"^(?:the|our|my|an?)\s+(\S.*)$")
_DIGIT_RUNS = re.compile(r"\d+")


# --------------------------------------------------------------------------
# Tier 0: normalization helpers
# --------------------------------------------------------------------------

def normalize_type(raw: Optional[str]) -> str:
    """Map an LLM-provided type onto the closed list; anything else -> Other."""
    return _TYPE_LOOKUP.get((raw or "").strip().lower(), "Other")


def make_key(name: str) -> str:
    """Matching key for an entity name.

    Lowercases, drops leading articles, and keeps meaningful symbols
    instead of deleting them (so C++, C# and C stay distinct):
        c++ -> cpp, c# -> csharp, .net -> dotnet, node.js -> nodejs.
    Remaining punctuation (hyphens, underscores, ...) becomes a space.
    """
    s = unicodedata.normalize("NFKC", name).strip().lower()

    while True:  # articles only when followed by whitespace, so "A/B testing" survives
        m = _ARTICLE_PREFIX.match(s)
        if not m:
            break
        s = m.group(1)

    s = re.sub(r"(?<=[a-z0-9])\+\+", "pp", s)
    s = re.sub(r"(?<=[a-z0-9])#", "sharp", s)
    s = re.sub(r"(?<=[a-z0-9])\+", "plus", s)
    s = re.sub(r"(?<![a-z0-9])\.(?=[a-z])", "dot", s)  # .net -> dotnet
    s = re.sub(r"(?<=[a-z])\.(?=[a-z])", "", s)  # node.js -> nodejs
    s = "".join(ch if ch.isalnum() else " " for ch in s)
    return " ".join(s.split())


def normalize_relation(raw: Optional[str]) -> str:
    """UPPER_SNAKE_CASE, safe to use as a Cypher relationship label."""
    s = re.sub(r"[^A-Za-z0-9]+", "_", (raw or "").strip()).strip("_").upper()
    if not s:
        return "RELATED_TO"
    if s[0].isdigit():
        s = f"REL_{s}"
    return s


def load_aliases(path: Path) -> Dict[str, str]:
    """Load a user-editable alias file: {"alias": "Canonical Name", ...}.
    Keys starting with "_" are ignored so the file can hold comments."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Alias file must be a JSON object of alias -> canonical name")
    return {str(k): str(v) for k, v in data.items() if not str(k).startswith("_")}


def _prepare_aliases(raw: Mapping[str, str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Returns (alias key -> target key, target key -> display name as written)."""
    key_map: Dict[str, str] = {}
    display: Dict[str, str] = {}
    for alias, target in raw.items():
        if str(alias).startswith("_"):
            continue
        ak, tk = make_key(alias), make_key(target)
        if ak and tk and ak != tk:
            key_map[ak] = tk
            display[tk] = " ".join(str(target).split())
    return key_map, display


def _keyify(surface: str, key_map: Mapping[str, str]) -> Tuple[str, bool]:
    """Key for a surface form after following the alias chain (cycle-safe)."""
    key = make_key(surface)
    via_alias = False
    for _ in range(10):
        nxt = key_map.get(key)
        if nxt is None or nxt == key:
            break
        key, via_alias = nxt, True
    return key, via_alias


# --------------------------------------------------------------------------
# Tier 2: pairwise comparison with guards
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class _Verdict:
    merge: bool
    score: float
    blocked_by: Optional[str] = None
    candidate: bool = False  # worth listing in the "possible duplicates" report


def _compare(a: str, b: str, threshold: float, min_len: int) -> _Verdict:
    if a == b:
        return _Verdict(True, 100.0)

    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) >= 3 and short in long_:
        return _Verdict(False, float(fuzz.ratio(a, b)), "substring", True)

    # plain ratio catches spacing variants ("scikit learn" / "scikitlearn");
    # token_sort catches word-order variants ("learning machine" / "machine learning")
    score = float(max(fuzz.ratio(a, b), fuzz.token_sort_ratio(a, b)))
    if score < threshold:
        return _Verdict(False, score)
    if min(len(a), len(b)) < min_len:
        return _Verdict(False, score, "short name", True)
    if _DIGIT_RUNS.findall(a) != _DIGIT_RUNS.findall(b):
        return _Verdict(False, score, "digits differ", True)
    return _Verdict(True, score)


# --------------------------------------------------------------------------
# Internal accumulators
# --------------------------------------------------------------------------

@dataclass
class _Group:
    """All mentions that share one matching key."""

    key: str
    surfaces: Counter = field(default_factory=Counter)
    types: Counter = field(default_factory=Counter)
    type_first_seen: Dict[str, int] = field(default_factory=dict)
    descriptions: List[str] = field(default_factory=list)
    sources: Dict[SourceRef, None] = field(default_factory=dict)
    alias_surfaces: set = field(default_factory=set)
    count: int = 0


@dataclass
class _Cluster:
    anchor: _Group
    members: List[Tuple[_Group, str, Optional[float]]]  # (group, tier, score)
    stub: bool = False
    extra_sources: Dict[SourceRef, None] = field(default_factory=dict)
    name: str = ""


def _collect_groups(
    extractions: Sequence[ChunkExtraction],
    key_map: Mapping[str, str],
    max_name_len: int,
    report: ResolutionReport,
) -> Dict[str, _Group]:
    groups: Dict[str, _Group] = {}
    seq = 0
    for ext in extractions:
        ref = SourceRef(source_file=ext.source_file, chunk_index=ext.chunk_index)
        for node in ext.payload.nodes:
            report.input_node_mentions += 1
            seq += 1
            surface = " ".join(node.name.split())
            if not surface or len(surface) > max_name_len:
                report.dropped_names.append(surface[:max_name_len] or repr(node.name))
                continue
            key, via_alias = _keyify(surface, key_map)
            if not key:
                report.dropped_names.append(surface)
                continue

            g = groups.setdefault(key, _Group(key=key))
            g.surfaces[surface] += 1
            g.count += 1
            g.sources.setdefault(ref, None)
            if via_alias:
                g.alias_surfaces.add(surface)

            norm_type = normalize_type(node.entity_type)
            raw_type = (node.entity_type or "").strip()
            if norm_type == "Other" and raw_type.lower() != "other":
                report.unmapped_types[raw_type or "(empty)"] = (
                    report.unmapped_types.get(raw_type or "(empty)", 0) + 1
                )
            g.types[norm_type] += 1
            g.type_first_seen.setdefault(norm_type, seq)

            if node.description and node.description.strip():
                g.descriptions.append(node.description.strip())
    return groups


def _cluster_groups(
    groups: Dict[str, _Group], threshold: float, min_len: int
) -> Tuple[List[_Cluster], List[Tuple[str, str, float, str]]]:
    """Greedy, deterministic clustering: most frequent keys anchor clusters."""
    ordered = sorted(groups.values(), key=lambda g: (-g.count, len(g.key), g.key))
    clusters: List[_Cluster] = []
    blocked: List[Tuple[str, str, float, str]] = []

    for g in ordered:
        best: Optional[Tuple[_Cluster, _Verdict]] = None
        for c in clusters:
            v = _compare(g.key, c.anchor.key, threshold, min_len)
            if v.merge:
                if best is None or v.score > best[1].score:
                    best = (c, v)
            elif v.candidate:
                blocked.append((g.key, c.anchor.key, v.score, v.blocked_by or ""))
        if best is not None:
            best[0].members.append((g, "fuzzy", best[1].score))
        else:
            clusters.append(_Cluster(anchor=g, members=[(g, "anchor", None)]))
    return clusters, blocked


def _canonical_name(cluster: _Cluster, alias_display: Mapping[str, str]) -> str:
    for g, _, _ in cluster.members:  # a user-chosen display name wins
        if g.key in alias_display:
            return alias_display[g.key]
    surfaces: Counter = Counter()
    for g, _, _ in cluster.members:
        surfaces.update(g.surfaces)
    # most frequent; ties -> prefer a form with capitals, then shorter, then alphabetical
    return sorted(
        surfaces.items(),
        key=lambda kv: (-kv[1], not any(ch.isupper() for ch in kv[0]), len(kv[0]), kv[0]),
    )[0][0]


def _build_node(cluster: _Cluster) -> ResolvedNode:
    surfaces: Counter = Counter()
    types: Counter = Counter()
    first_seen: Dict[str, int] = {}
    descriptions: List[str] = []
    sources: Dict[SourceRef, None] = {}
    count = 0
    for g, _, _ in cluster.members:
        surfaces.update(g.surfaces)
        types.update(g.types)
        for t, idx in g.type_first_seen.items():
            first_seen[t] = min(idx, first_seen.get(t, idx))
        descriptions.extend(g.descriptions)
        sources.update(g.sources)
        count += g.count
    sources.update(cluster.extra_sources)

    if types:
        entity_type = sorted(types, key=lambda t: (-types[t], first_seen[t]))[0]
    else:
        entity_type = "Other"

    return ResolvedNode(
        name=cluster.name,
        entity_type=entity_type,
        description=max(descriptions, key=len) if descriptions else None,
        aliases=sorted(s for s in surfaces if s != cluster.name),
        sources=list(sources),
        mention_count=count,
        inferred=cluster.stub,
    )


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def resolve(
    extractions: Sequence[ChunkExtraction],
    *,
    aliases: Optional[Mapping[str, str]] = None,
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
    min_fuzzy_len: int = DEFAULT_MIN_FUZZY_LEN,
    max_name_len: int = DEFAULT_MAX_NAME_LEN,
) -> Tuple[ResolvedGraph, ResolutionReport]:
    """Resolve pooled chunk extractions into a canonical graph + report."""
    report = ResolutionReport()
    key_map, alias_display = _prepare_aliases(aliases or {})

    # ---- nodes: group by key, then cluster conservatively -------------------
    groups = _collect_groups(extractions, key_map, max_name_len, report)
    clusters, blocked_raw = _cluster_groups(groups, fuzzy_threshold, min_fuzzy_len)

    used_names: set = set()

    def assign_name(cluster: _Cluster, preferred: str) -> None:
        name = preferred
        if name in used_names:  # safety net: never let two clusters share a name
            name = f"{preferred} [{cluster.anchor.key}]"
        used_names.add(name)
        cluster.name = name

    key_index: Dict[str, _Cluster] = {}
    for c in clusters:
        assign_name(c, _canonical_name(c, alias_display))
        for g, _, _ in c.members:
            key_index[g.key] = c

    # ---- edges: remap endpoints, create stubs for dangling ones ------------
    def lookup(surface: str, ref: SourceRef) -> Optional[_Cluster]:
        key, _ = _keyify(surface, key_map)
        if not key:
            return None
        cluster = key_index.get(key)
        if cluster is None:
            best: Optional[Tuple[_Cluster, float]] = None
            for c in clusters:
                v = _compare(key, c.anchor.key, fuzzy_threshold, min_fuzzy_len)
                if v.merge and (best is None or v.score > best[1]):
                    best = (c, v.score)
            if best is not None:
                cluster = best[0]
            else:  # dangling endpoint -> flagged stub node
                for c in clusters:
                    v = _compare(key, c.anchor.key, fuzzy_threshold, min_fuzzy_len)
                    if v.candidate:
                        blocked_raw.append((key, c.anchor.key, v.score, v.blocked_by or ""))
                stub_group = _Group(key=key)
                stub_group.surfaces[surface] += 1
                cluster = _Cluster(anchor=stub_group, members=[(stub_group, "anchor", None)], stub=True)
                assign_name(cluster, surface)
                clusters.append(cluster)
                report.stub_nodes.append(cluster.name)
            key_index[key] = cluster
        if cluster.stub:
            cluster.extra_sources.setdefault(ref, None)
        return cluster

    edge_acc: Dict[Tuple[str, str, str], Dict] = {}
    for ext in extractions:
        ref = SourceRef(source_file=ext.source_file, chunk_index=ext.chunk_index)
        for edge in ext.payload.edges:
            report.input_edge_mentions += 1
            src_s = " ".join(edge.source.split())
            tgt_s = " ".join(edge.target.split())
            if not src_s or not tgt_s or len(src_s) > max_name_len or len(tgt_s) > max_name_len:
                report.dropped_edges += 1
                continue
            a = lookup(src_s, ref)
            b = lookup(tgt_s, ref)
            if a is None or b is None:
                report.dropped_edges += 1
                continue
            if a is b:
                report.self_loops_dropped += 1
                continue
            rel = normalize_relation(edge.relationship_type)
            acc = edge_acc.setdefault(
                (a.name, rel, b.name), {"count": 0, "context": None, "sources": {}}
            )
            acc["count"] += 1
            acc["sources"].setdefault(ref, None)
            if acc["context"] is None and edge.context and edge.context.strip():
                acc["context"] = edge.context.strip()

    # ---- assemble output -----------------------------------------------------
    nodes = sorted(
        (_build_node(c) for c in clusters),
        key=lambda n: (-n.mention_count, n.name.lower()),
    )
    edges = sorted(
        (
            ResolvedEdge(
                source=s,
                target=t,
                relationship_type=r,
                context=acc["context"],
                sources=list(acc["sources"]),
                evidence_count=acc["count"],
            )
            for (s, r, t), acc in edge_acc.items()
        ),
        key=lambda e: (-e.evidence_count, e.source.lower(), e.relationship_type, e.target.lower()),
    )

    # ---- report ----------------------------------------------------------------
    for c in clusters:
        if c.stub:
            continue
        for g, tier, score in c.members:
            for surface in sorted(g.surfaces):
                if surface == c.name:
                    continue
                t = tier
                if tier == "anchor":
                    t = "alias" if surface in g.alias_surfaces else "exact"
                report.merges.append(
                    MergeRecord(
                        canonical=c.name,
                        merged_from=surface,
                        tier=t,
                        score=round(score, 1) if score is not None else None,
                    )
                )

    seen_pairs: set = set()
    for key_a, key_b, score, reason in sorted(blocked_raw, key=lambda r: -r[2]):
        ca, cb = key_index.get(key_a), key_index.get(key_b)
        if ca is None or cb is None or ca is cb:
            continue
        pair = frozenset((ca.name, cb.name))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        report.blocked_pairs.append(
            BlockedPair(a=ca.name, b=cb.name, score=round(score, 1), blocked_by=reason)
        )

    report.output_nodes = len(nodes)
    report.output_edges = len(edges)
    return ResolvedGraph(nodes=nodes, edges=edges), report