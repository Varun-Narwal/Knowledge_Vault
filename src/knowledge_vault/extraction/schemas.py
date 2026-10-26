import copy
from typing import List, Optional
from pydantic import BaseModel, Field

# Single source of truth for the closed entity-type list. The extractor's
# output schema, the resolver's type normalization and the prompt all use it.
ALLOWED_ENTITY_TYPES = (
    "Person",
    "Organization",
    "Tool",
    "Framework",
    "Language",
    "Concept",
    "Project",
    "Other",
)


class ExtractedNode(BaseModel):
    name: str = Field(..., description="The canonical name of the entity, concept, or tool")
    entity_type: str = Field(..., description="Broad category, e.g., Person, Framework, Concept, Location")
    description: Optional[str] = Field(default=None, description="A brief summary of what this entity is based on the text")

class ExtractedEdge(BaseModel):
    source: str = Field(..., description="The name of the source node (must match an extracted node's name)")
    target: str = Field(..., description="The name of the target node (must match an extracted node's name)")
    relationship_type: str = Field(..., description="A concise verb phrase describing how they connect (e.g., USES, DEPENDS_ON, AUTHORED)")
    context: Optional[str] = Field(default=None, description="The exact sentence or snippet from the text that proves this relationship")

class ExtractionGraphPayload(BaseModel):
    nodes: List[ExtractedNode] = Field(default_factory=list)
    edges: List[ExtractedEdge] = Field(default_factory=list)


def _inline_refs(node, defs):
    """Replace $ref pointers with the definitions they point to, so the
    schema handed to Ollama is fully self-contained."""
    if isinstance(node, dict):
        if "$ref" in node:
            name = node["$ref"].rsplit("/", 1)[-1]
            return _inline_refs(copy.deepcopy(defs[name]), defs)
        return {k: _inline_refs(v, defs) for k, v in node.items()}
    if isinstance(node, list):
        return [_inline_refs(v, defs) for v in node]
    return node


def build_llm_output_schema() -> dict:
    """JSON schema used to constrain Ollama's decoding (`format` field).

    Derived from the Pydantic models, with two deliberate differences:
    entity_type is an enum of the closed type list, and both top-level
    arrays are required so the model always emits them.
    """
    schema = ExtractionGraphPayload.model_json_schema()
    defs = schema.pop("$defs", {})
    schema = _inline_refs(schema, defs)
    node_props = schema["properties"]["nodes"]["items"]["properties"]
    node_props["entity_type"] = {"type": "string", "enum": list(ALLOWED_ENTITY_TYPES)}
    schema["required"] = ["nodes", "edges"]
    return schema