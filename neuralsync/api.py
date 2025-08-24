from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class MemoryIn(BaseModel):
    text: str
    kind: str = 'fact'
    scope: str = 'global'
    tool: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    confidence: float = 0.8
    benefit: Optional[float] = None
    consistency: Optional[float] = None
    ttl_ms: Optional[int] = None
    source: Optional[str] = None
    meta: Optional[Dict[str,Any]] = None

class RecallIn(BaseModel):
    query: str
    top_k: int = 8
    scope: str = 'any'
    tool: Optional[str] = None
    use_embedding: bool = True

class PersonaIn(BaseModel):
    text: str
