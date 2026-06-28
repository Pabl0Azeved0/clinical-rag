from __future__ import annotations

from clinical_rag.domain.models import Chunk, Document


def chunk_document(doc: Document, chunk_size: int, overlap: int) -> list[Chunk]:
    """Split a Document into Chunks using paragraph-aware character chunking with overlap.

    Prefers paragraph breaks (double newline) as split points before falling back to
    hard character limits, ensuring chunk boundaries align with natural text structure
    where possible.
    """
    text = doc.text
    if not text:
        return []

    # Split on paragraph boundaries first, then reassemble greedily
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Reassemble paragraphs into chunks respecting chunk_size
    raw_chunks: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = (current + "\n\n" + para).strip() if current else para
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                raw_chunks.append(current)
            # Para itself might exceed chunk_size — hard-split it
            while len(para) > chunk_size:
                raw_chunks.append(para[:chunk_size])
                para = para[chunk_size - overlap :]
            current = para
    if current:
        raw_chunks.append(current)

    # Apply overlap between consecutive chunks
    chunks: list[Chunk] = []
    for i, chunk_text in enumerate(raw_chunks):
        # Prepend tail of previous chunk for context overlap
        if i > 0 and overlap > 0:
            prev = raw_chunks[i - 1]
            prefix = prev[-overlap:].strip()
            chunk_text = (prefix + " " + chunk_text).strip()

        chunks.append(
            Chunk(
                id=f"{doc.id}-{i}",
                doc_id=doc.id,
                title=doc.title,
                url=doc.url,
                text=chunk_text,
                chunk_index=i,
            )
        )

    return chunks


def chunk_documents(docs: list[Document], chunk_size: int, overlap: int) -> list[Chunk]:
    """Chunk all documents and return a flat list of Chunks."""
    result: list[Chunk] = []
    for doc in docs:
        result.extend(chunk_document(doc, chunk_size, overlap))
    return result
