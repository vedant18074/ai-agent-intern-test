from dataclasses import dataclass
from typing import List, Tuple

from app.models import DocumentChunk
from app.retrieval.retriever import RetrievedChunk


@dataclass
class EvidenceResult:
    chunks: List[RetrievedChunk]
    conflicts: List[Tuple[RetrievedChunk, RetrievedChunk]]
    excluded_chunks: List[RetrievedChunk]

    @property
    def has_conflict(self) -> bool:
        return len(self.conflicts) > 0

    @property
    def requires_handoff(self) -> bool:
        return self.has_conflict


class EvidenceAnalyzer:
    """
    Converts retrieval results into safe evidence for the agent.

    Responsibilities:
    - Remove internal-only content from customer evidence.
    - Prefer active and official sources.
    - Preserve multiple authoritative sources when they disagree.
    - Detect simple, deterministic policy contradictions.
    """

    def analyze(
        self,
        retrieved: List[RetrievedChunk],
    ) -> EvidenceResult:

        customer_chunks = []
        excluded_chunks = []

        for result in retrieved:
            if self._is_customer_safe(result.chunk):
                customer_chunks.append(result)
            else:
                excluded_chunks.append(result)

        conflicts = self._detect_conflicts(
            customer_chunks
        )

        return EvidenceResult(
            chunks=customer_chunks,
            conflicts=conflicts,
            excluded_chunks=excluded_chunks,
        )

    @staticmethod
    def _is_customer_safe(
        chunk: DocumentChunk,
    ) -> bool:

        audience = (
            chunk.audience or ""
        ).strip().lower()

        return audience != "internal"

    def _detect_conflicts(
        self,
        chunks: List[RetrievedChunk],
    ) -> List[Tuple[RetrievedChunk, RetrievedChunk]]:

        conflicts = []

        for index, first in enumerate(chunks):
            for second in chunks[index + 1:]:
                if self._is_conflicting(
                    first.chunk,
                    second.chunk,
                ):
                    conflicts.append(
                        (first, second)
                    )

        return conflicts

    @staticmethod
    def _is_conflicting(
        first: DocumentChunk,
        second: DocumentChunk,
    ) -> bool:

        # Only compare current authoritative sources.
        if (
            first.status.lower() != "active"
            or second.status.lower() != "active"
        ):
            return False

        if (
            first.policy_authority.lower()
            != "official"
            or second.policy_authority.lower()
            != "official"
        ):
            return False

        first_text = first.content.lower()
        second_text = second.content.lower()

        # Deterministic contradiction:
        # hand-wash vs dishwasher-safe.
        hand_wash_first = (
            "hand-wash" in first_text
            or "hand wash" in first_text
            or "hand-washed" in first_text
        )

        hand_wash_second = (
            "hand-wash" in second_text
            or "hand wash" in second_text
            or "hand-washed" in second_text
        )

        dishwasher_first = (
            "dishwasher safe" in first_text
            or "dishwasher-safe" in first_text
        )

        dishwasher_second = (
            "dishwasher safe" in second_text
            or "dishwasher-safe" in second_text
        )

        if (
            hand_wash_first
            and dishwasher_second
        ):
            return True

        if (
            dishwasher_first
            and hand_wash_second
        ):
            return True

        return False