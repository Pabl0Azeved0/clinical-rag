from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Document:
    id: str
    title: str
    url: str
    text: str
    also_called: list[str] = field(default_factory=list)


@dataclass
class Chunk:
    id: str
    doc_id: str
    title: str
    url: str
    text: str
    chunk_index: int
