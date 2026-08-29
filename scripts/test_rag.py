from app.rag import RAGService


rag = RAGService("knowledge-base")


query = "What is the return policy?"


evidence = rag.retrieve(
    query=query,
    top_k=5,
)


print("\nQUESTION:")
print(query)

print("\nRETRIEVED EVIDENCE:")

for result in evidence.chunks:
    chunk = result.chunk

    print("\n------------------------------")
    print(f"Score: {result.score:.4f}")
    print(f"Source: {chunk.source_file}")
    print(f"Title: {chunk.title}")
    print(f"Heading: {chunk.heading}")
    print(f"Status: {chunk.status}")
    print(f"Audience: {chunk.audience}")
    print(f"Authority: {chunk.policy_authority}")
    print("\nContent:")
    print(chunk.content[:500])

print("\n------------------------------")

print(
    f"\nConflicts detected: {len(evidence.conflicts)}"
)

print(
    f"Internal chunks excluded: "
    f"{len(evidence.excluded_chunks)}"
)