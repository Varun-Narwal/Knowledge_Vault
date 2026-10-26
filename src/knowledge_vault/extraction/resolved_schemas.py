from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from knowledge_vault.extraction.schemas import ExtractionGraphPayload


class SourceRef(BaseModel):
    """Where a mention came from. Frozen so it can be used as a dict key."""

    model_config = ConfigDict(frozen=True)

    source_file: str
    chunk_index: int


class ChunkExtraction(BaseModel):
    """One chunk's raw LLM output, tagged by the pipeline (not the LLM)
    with the chunk it came from."""

    source_file: str
    chunk_index: int
    payload: ExtractionGraphPayload


class ResolvedNode(BaseModel):
    name: str
    entity_type: str
    description: Optional[str] = None
    aliases: List[str] = Field(default_factory=list)
    sources: List[SourceRef] = Field(default_factory=list)
    mention_count: int = 0
    inferred: bool = False  # True for stub nodes created from dangling edge endpoints


class ResolvedEdge(BaseModel):
    source: str
    target: str
    relationship_type: str
    context: Optional[str] = None
    sources: List[SourceRef] = Field(default_factory=list)
    evidence_count: int = 1


class ResolvedGraph(BaseModel):
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    nodes: List[ResolvedNode] = Field(default_factory=list)
    edges: List[ResolvedEdge] = Field(default_factory=list)


class MergeRecord(BaseModel):
    canonical: str
    merged_from: str
    tier: str  # "exact" | "alias" | "fuzzy"
    score: Optional[float] = None


class BlockedPair(BaseModel):
    a: str
    b: str
    score: float
    blocked_by: str  # "substring" | "digits differ" | "short name"


class ResolutionReport(BaseModel):
    input_node_mentions: int = 0
    input_edge_mentions: int = 0
    output_nodes: int = 0
    output_edges: int = 0
    merges: List[MergeRecord] = Field(default_factory=list)
    blocked_pairs: List[BlockedPair] = Field(default_factory=list)
    dropped_names: List[str] = Field(default_factory=list)
    dropped_edges: int = 0
    self_loops_dropped: int = 0
    stub_nodes: List[str] = Field(default_factory=list)
    unmapped_types: Dict[str, int] = Field(default_factory=dict)