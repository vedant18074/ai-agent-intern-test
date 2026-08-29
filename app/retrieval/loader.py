from pathlib import Path
from typing import List

from app.models import Document


class KnowledgeBaseLoader:
    def __init__(self, knowledge_base_path: str):
        self.knowledge_base_path = Path(knowledge_base_path)

    def load_documents(self) -> List[Document]:
        documents = []

        for file_path in sorted(self.knowledge_base_path.glob("*.md")):
            document = self._load_document(file_path)
            documents.append(document)

        return documents

    def _load_document(self, file_path: Path) -> Document:
        content = file_path.read_text(encoding="utf-8")

        metadata, body = self._parse_front_matter(content)

        return Document(
            document_id=metadata.get("document_id", ""),
            title=metadata.get("title", file_path.stem),
            content=body,
            source_file=file_path.name,
            status=metadata.get("status"),
            audience=metadata.get("audience"),
            policy_authority=metadata.get("policy_authority"),
            effective_date=metadata.get("effective_date"),
            last_reviewed=metadata.get("last_reviewed"),
            supersedes=metadata.get("supersedes"),
            superseded_by=metadata.get("superseded_by"),
        )

    @staticmethod
    def _parse_front_matter(content: str):
        if not content.startswith("---"):
            return {}, content

        parts = content.split("---", 2)

        if len(parts) != 3:
            return {}, content

        front_matter = parts[1].strip()
        body = parts[2].strip()

        metadata = {}

        for line in front_matter.splitlines():
            if ":" not in line:
                continue

            key, value = line.split(":", 1)

            metadata[key.strip()] = value.strip()

        return metadata, body