from typing import List

from app.models import Document, DocumentChunk


class DocumentChunker:
    def __init__(self, max_chunk_size: int = 1200):
        self.max_chunk_size = max_chunk_size

    def chunk_document(self, document: Document) -> List[DocumentChunk]:
        sections = self._split_into_sections(document.content)

        chunks = []

        for index, (heading, content) in enumerate(sections):
            section_chunks = self._split_large_section(content)

            for sub_index, chunk_content in enumerate(section_chunks):
                chunk_id = f"{document.document_id}-{index}-{sub_index}"

                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=document.document_id,
                        content=chunk_content,
                        source_file=document.source_file,
                        title=document.title,
                        heading=heading,
                        status=document.status,
                        audience=document.audience,
                        policy_authority=document.policy_authority,
                        effective_date=document.effective_date,
                    )
                )

        return chunks

    def _split_into_sections(self, content: str):
        lines = content.splitlines()

        sections = []
        current_heading = None
        current_lines = []

        for line in lines:
            stripped = line.strip()

            if stripped.startswith("#"):
                if current_lines:
                    sections.append(
                        (
                            current_heading,
                            "\n".join(current_lines).strip(),
                        )
                    )

                current_heading = stripped.lstrip("#").strip()
                current_lines = []
            else:
                current_lines.append(line)

        if current_lines:
            sections.append(
                (
                    current_heading,
                    "\n".join(current_lines).strip(),
                )
            )

        return [
            (heading, content)
            for heading, content in sections
            if content
        ]

    def _split_large_section(self, content: str) -> List[str]:
        if len(content) <= self.max_chunk_size:
            return [content]

        paragraphs = content.split("\n\n")

        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()

            if not paragraph:
                continue

            if (
                current_chunk
                and len(current_chunk) + len(paragraph) + 2
                > self.max_chunk_size
            ):
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks