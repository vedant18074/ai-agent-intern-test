from dataclasses import dataclass

from app.retrieval.retriever import RetrievedChunk


@dataclass
class Citation:
    source_file: str
    heading: str
    document_id: str
    score: float

    def format(self) -> str:
        return (
            f"Source: {self.source_file} | "
            f"Section: {self.heading or 'General'}"
        )


class CitationBuilder:
    @staticmethod
    def build(
        result: RetrievedChunk,
    ) -> Citation:
        chunk = result.chunk

        return Citation(
            source_file=chunk.source_file,
            heading=chunk.heading or "",
            document_id=chunk.document_id,
            score=result.score,
        )

    @staticmethod
    def build_many(
        results: list[RetrievedChunk],
    ) -> list[Citation]:
        return [
            CitationBuilder.build(result)
            for result in results
        ]