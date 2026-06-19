"""
ingest.py

Pipeline: PDF -> Markdown -> chunks -> embeddings -> ChromaDB

Run this once (or whenever you change chunking logic) to (re)build the vector store:
    python ingest.py --pdf data/textbook.pdf
"""

import argparse
import chromadb


def pdf_to_markdown(pdf_path: str) -> str:
    """
    Convert PDF to markdown text using pymupdf4llm.

    Why markdown instead of raw text: preserves heading structure (Chapter X,
    Section X.Y) which the chunking step below relies on to keep chunks
    conceptually coherent instead of splitting mid-theorem/mid-proof.

    TODO:
    - Load the PDF with pymupdf4llm
    - Return the extracted markdown string

    Docs: https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/
    """
    raise NotImplementedError


def chunk_markdown(markdown_text: str) -> list[dict]:
    """
    Split markdown into chunks, ideally aligned to section boundaries.

    Decision you need to make and be ready to defend in an interview:
    - Chunk by markdown heading (## Section X.Y) vs. fixed character/token window
    - What chunk_overlap (if any) you use, and why
    - What metadata to attach to each chunk (e.g., chapter, section number,
      page range) so you can cite sources later

    Each returned dict should look something like:
        {"text": "...", "metadata": {"section": "3.2", "chapter": "Loop Invariants"}}

    TODO:
    - Implement section-aware splitting (try splitting on lines starting with '#')
    - Fall back to a fixed-size splitter for sections that are still too long
    - Return list of chunk dicts

    Reference: https://www.pinecone.io/learn/chunking-strategies/
    """
    raise NotImplementedError


def embed_chunks(chunks: list[dict]) -> list[list[float]]:
    """
    Generate embeddings for each chunk's text.

    Decision to make: which embedding model/provider (Anthropic embeddings,
    sentence-transformers locally, etc.) and why — cost, latency, and quality
    tradeoffs are all fair game in an interview.

    TODO:
    - Pick an embedding approach
    - Return a list of embedding vectors, same order as `chunks`
    """
    raise NotImplementedError


def store_in_chroma(chunks: list[dict], embeddings: list[list[float]], collection_name: str = "textbook"):
    """
    Persist chunks + embeddings + metadata into a local ChromaDB collection.

    TODO:
    - Create/get a persistent ChromaDB client (see chroma_db/ dir)
    - Create or reset the collection
    - Add documents, embeddings, metadatas, and ids in batches

    Docs: https://docs.trychroma.com/getting-started
    """
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description="Ingest a textbook PDF into ChromaDB")
    parser.add_argument("--pdf", required=True, help="Path to textbook PDF")
    parser.add_argument("--collection", default="textbook", help="ChromaDB collection name")
    args = parser.parse_args()

    print(f"Parsing {args.pdf}...")
    markdown_text = pdf_to_markdown(args.pdf)

    print("Chunking...")
    chunks = chunk_markdown(markdown_text)
    print(f"  -> {len(chunks)} chunks")

    print("Embedding...")
    embeddings = embed_chunks(chunks)

    print("Storing in ChromaDB...")
    store_in_chroma(chunks, embeddings, collection_name=args.collection)

    print("Done.")


if __name__ == "__main__":
    main()
