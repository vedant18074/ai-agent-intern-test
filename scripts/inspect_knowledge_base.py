from pathlib import Path

from app.retrieval.loader import KnowledgeBaseLoader


def main():
    project_root = Path(__file__).resolve().parents[1]
    knowledge_base_path = project_root / "knowledge-base"

    loader = KnowledgeBaseLoader(str(knowledge_base_path))
    documents = loader.load_documents()

    print(f"\nLoaded {len(documents)} documents\n")
    print("=" * 80)

    for document in documents:
        print(f"File:              {document.source_file}")
        print(f"Document ID:       {document.document_id}")
        print(f"Title:             {document.title}")
        print(f"Status:            {document.status}")
        print(f"Audience:          {document.audience}")
        print(f"Authority:         {document.policy_authority}")
        print(f"Effective date:    {document.effective_date}")
        print(f"Last reviewed:     {document.last_reviewed}")
        print(f"Supersedes:        {document.supersedes}")
        print(f"Superseded by:     {document.superseded_by}")
        print(f"Content length:    {len(document.content)} characters")
        print("-" * 80)


if __name__ == "__main__":
    main()