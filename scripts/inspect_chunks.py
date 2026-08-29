from pathlib import Path
from collections import Counter

from app.retrieval.loader import KnowledgeBaseLoader
from app.retrieval.chunker import DocumentChunker


def main():
    project_root = Path(__file__).resolve().parents[1]
    knowledge_base_path = project_root / "knowledge-base"

    loader = KnowledgeBaseLoader(str(knowledge_base_path))
    documents = loader.load_documents()

    chunker = DocumentChunker(max_chunk_size=1200)

    all_chunks = []

    for document in documents:
        chunks = chunker.chunk_document(document)
        all_chunks.extend(chunks)

    print(f"\nDocuments loaded: {len(documents)}")
    print(f"Total chunks:     {len(all_chunks)}")

    print("\n" + "=" * 80)
    print("CHUNKS PER DOCUMENT")
    print("=" * 80)

    chunk_counts = Counter(
        chunk.source_file for chunk in all_chunks
    )

    for source_file, count in chunk_counts.items():
        print(f"{source_file:<50} {count} chunks")

    print("\n" + "=" * 80)
    print("CHUNK SIZE STATISTICS")
    print("=" * 80)

    sizes = [len(chunk.content) for chunk in all_chunks]

    print(f"Smallest chunk:    {min(sizes)} characters")
    print(f"Largest chunk:     {max(sizes)} characters")
    print(f"Average chunk:     {sum(sizes) / len(sizes):.2f} characters")

    print("\n" + "=" * 80)
    print("SAMPLE CHUNKS")
    print("=" * 80)

    for chunk in all_chunks[:5]:
        print(f"\nChunk ID:     {chunk.chunk_id}")
        print(f"Source:      {chunk.source_file}")
        print(f"Heading:     {chunk.heading}")
        print(f"Status:      {chunk.status}")
        print(f"Audience:    {chunk.audience}")
        print(f"Authority:   {chunk.policy_authority}")
        print(f"Size:        {len(chunk.content)} characters")
        print("\nContent:")
        print(chunk.content[:500])
        print("-" * 80)


if __name__ == "__main__":
    main()