"""
ingest.py

Pipeline: PDF -> Markdown -> chunks -> embeddings -> ChromaDB

Run this once (or whenever you change chunking logic) to (re)build the vector store:
    python ingest.py --pdf data/textbook.pdf
"""

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
import torch

from pathlib import Path
import argparse
import re
import chromadb

############ MARKDOWN CONVERSION ############


# Load model once, if needed can reuse across PDF
device = "mps" if torch.backends.mps.is_available() else "cpu"
_model_dict = create_model_dict()
_converter = PdfConverter(artifact_dict=_model_dict)

def pdf_to_markdown(pdf_path: str) -> tuple[str, dict]:
    """
    Convert PDF to markdown text.
        Using Marker.

    TODO:
    - Load the PDF with Nougat
    - Return the extracted markdown string
    """
    rendered = _converter(pdf_path)
    text, metadata, images = text_from_rendered(rendered)
    
    if not text.strip():
        raise RuntimeError(f"Marker produced empty output for {pdf_path}")
    
    return text, images

def save_markdown_output(pdf_path: str, markdown_text: str, markdown_image: dict) -> Path:
    """
    Write markdown text and extracted images to output/<pdf_stem>/.
    """
    output_dir = Path("output") / Path(pdf_path).stem
    output_dir.mkdir(parents=True, exist_ok=True)

    md_path = output_dir / "output.md"
    md_path.write_text(markdown_text, encoding="utf-8")
    print(f"  Markdown written to {md_path}")

    if markdown_image:
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)
        for filename, image in markdown_image.items():
            image.save(images_dir / filename)
        print(f"  {len(markdown_image)} image(s) written to {images_dir}")
    else:
        print("  No images extracted.")

    return output_dir

############ CHUNK MARKDOWN ############

# Matches headings like "# 1. Propositional Logic", "# 1.2 Logical Operations",
# "# 1.2.1 Negation (¬)". Captures the dotted number and the title separately.
_HEADING_RE = re.compile(
    r"^#+\s+(?P<num>\d+(?:\.\d+)*)\.?\s+(?P<title>.+?)\s*$",
    re.MULTILINE,
)

# Matches the start of a labeled block: "Definition 1.1 — Proposition.",
# "Example 1.2", "Lemma 1.1 — De Morgan's Laws."
_BLOCK_RE = re.compile(
    r"^(?:#+\s*)?(?P<kind>Definition|Example|Lemma|Theorem|Corollary|Proposition)\s+"
    r"(?P<label>\d+(?:\.\d+)*)\b",
    re.MULTILINE,
)

# Soft cap: no split mid-block, but flag/keep sections whole even if large.
_SOFT_CAP_CHARS = 2000


def _classify(num: str) -> tuple[str, str]:
    """Return (chapter_num, section_num) context from a dotted heading number."""
    parts = num.split(".")
    chapter = parts[0]
    section = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
    return chapter, section


def chunk_markdown(markdown_text: str) -> list[dict]:
    """
    Split markdown into chunks at the subsection level (e.g. 1.2.1, 1.2.2),
    additionally extracting labeled blocks (Definitions, Examples, Lemmas, etc.)
    as their own standalone chunks.

    TODO:
    - Find every numbered heading and slice the text between consecutive headings.
    - Each slice becomes one chunk, tagged with chapter/section/subsection metadata.
    - Within each slice, also emit a standalone chunk for each labeled block.
    - Soft cap: section boundaries are never broken; oversized chunks are kept
      whole but flagged in metadata so you can decide later.

    Returns a list of dicts: {"text": ..., "metadata": {...}}.
    """
    chunks: list[dict] = []

    # 1. Find all heading positions.
    headings = list(_HEADING_RE.finditer(markdown_text))
    if not headings:
        # Fallback
        return [{
            "text": markdown_text.strip(),
            "metadata": {"type": "section", "section": None, "chapter": None},
        }]

    # Track the most recent chapter title so subsections can inherit it.
    chapter_titles: dict[str, str] = {}

    # 2. Walk consecutive heading pairs, slicing the body between them.
    for i, h in enumerate(headings):
        num = h.group("num")
        title = h.group("title").strip()
        start = h.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(markdown_text)
        body = markdown_text[start:end].strip()

        chapter, section = _classify(num)

        # Record chapter titles (e.g. "# 1. Propositional Logic").
        if "." not in num:
            chapter_titles[chapter] = title

        depth = num.count(".") + 1  # 1 = chapter, 2 = section, 3 = subsection

        # Skip pure chapter headers with no body of their own (their content
        # lives in subsections), but keep section/subsection slices.
        if body:
            section_chunk = {
                "text": f"# {num} {title}\n\n{body}",
                "metadata": {
                    "type": "subsection" if depth >= 3 else ("section" if depth == 2 else "chapter"),
                    "number": num,
                    "title": title,
                    "chapter": chapter,
                    "chapter_title": chapter_titles.get(chapter),
                    "section": section,
                    "oversized": len(body) > _SOFT_CAP_CHARS,
                },
            }
            chunks.append(section_chunk)

        # 3. Extract labeled blocks within this slice as standalone chunks.
        block_matches = list(_BLOCK_RE.finditer(body))
        for j, b in enumerate(block_matches):
            b_start = b.start()
            b_end = block_matches[j + 1].start() if j + 1 < len(block_matches) else len(body)
            block_text = body[b_start:b_end].strip()

            chunks.append({
                "text": block_text,
                "metadata": {
                    "type": b.group("kind").lower(),   # "definition", "example", ...
                    "label": b.group("label"),          # "1.1", "1.2", ...
                    "number": num,
                    "title": title,
                    "chapter": chapter,
                    "chapter_title": chapter_titles.get(chapter),
                    "section": section,
                },
            })

    return chunks

############ CHUNK EMBEDDING ############

def embed_chunks(chunks: list[dict]) -> list[list[float]]:
    """
    Generate embeddings using nomic-embed-text-v1 (8192-token context window).

    TODO:
    - Prefixes each chunk with "search_document: " as required by the model.
    - Returns normalized vectors for cosine similarity in ChromaDB.
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True, device=device)
    model.max_seq_length = 2048
    texts = [f"search_document: {c['text']}" for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=8, normalize_embeddings=True)
    return embeddings.tolist()


def store_in_chroma(chunks: list[dict], embeddings: list[list[float]], collection_name: str = "textbook"):
    """
    Persist chunks + embeddings + metadata into a local ChromaDB collection.
    Resets the collection on each run so re-ingesting always produces a clean state.
    """
    client = chromadb.PersistentClient(path="chroma_db")

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # ChromaDB metadata values must be str/int/float/bool — strip None
    metadatas = [
        {k: v for k, v in c["metadata"].items() if v is not None}
        for c in chunks
    ]

    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=[c["text"] for c in chunks],
        embeddings=embeddings,
        metadatas=metadatas,
    )
    print(f"  Stored {len(chunks)} chunks in collection '{collection_name}'")


def main():
    parser = argparse.ArgumentParser(description="Ingest a textbook PDF into ChromaDB")
    parser.add_argument("--pdf", required=True, help="Path to textbook PDF")
    parser.add_argument("--collection", default="textbook", help="ChromaDB collection name")
    args = parser.parse_args()

    cached_md = Path("output") / Path(args.pdf).stem / "output.md"
    if cached_md.exists():
        print(f"Loading cached markdown from {cached_md}...")
        markdown_text = cached_md.read_text(encoding="utf-8")
    else:
        print(f"Parsing {args.pdf}...")
        markdown_text, markdown_image = pdf_to_markdown(args.pdf)
        save_markdown_output(args.pdf, markdown_text, markdown_image)

    print("Chunking...")
    chunks = chunk_markdown(markdown_text)
    sizes = [len(c["text"]) for c in chunks]
    print(f"  -> {len(chunks)} chunks | max={max(sizes)} min={min(sizes)} avg={sum(sizes)//len(sizes)} chars")

    print("Embedding...")
    embeddings = embed_chunks(chunks)

    print("Storing in ChromaDB...")
    store_in_chroma(chunks, embeddings, collection_name=args.collection)

    print("Done.")


if __name__ == "__main__":
    main()
