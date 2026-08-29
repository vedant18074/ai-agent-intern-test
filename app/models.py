from dataclasses import dataclass
from typing import Optional


@dataclass
class Document:
    document_id: str
    title: str
    content: str
    source_file: str

    status: Optional[str] = None
    audience: Optional[str] = None
    policy_authority: Optional[str] = None
    effective_date: Optional[str] = None
    last_reviewed: Optional[str] = None
    supersedes: Optional[str] = None
    superseded_by: Optional[str] = None

@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    content: str
    source_file: str
    title: str

    heading: Optional[str] = None
    status: Optional[str] = None
    audience: Optional[str] = None
    policy_authority: Optional[str] = None
    effective_date: Optional[str] = None