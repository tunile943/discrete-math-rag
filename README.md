# Discrete Structures RAG

This RAG system is a personal project intentionally tailored to "Lectures in Discrete Mathematics" by Matthew Eichhorn for Cornell CS 2800 rather than a generic PDF ingestion pipeline. This allows for structure-aware chunking (by theorem/definition/section) and manual validation of math-heavy extraction quality against known ground truth.

### Built With

* [![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
* [![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
* [![Anthropic Claude](https://img.shields.io/badge/Anthropic%20Claude-191919?style=for-the-badge&logo=anthropic&logoColor=white)](https://www.anthropic.com/)
* [![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B35?style=for-the-badge&logoColor=white)](https://www.trychroma.com/)
* [![Marker](https://img.shields.io/badge/Marker-2E86AB?style=for-the-badge&logoColor=white)](https://github.com/datalab-to/marker)

## Setup

```bash
# Marker 1.8.0 must be installed from GitHub (not on PyPI at this version)
pip install git+https://github.com/datalab-to/marker.git@v1.8.0

pip install -r requirements.txt
cp .env.example .env
```

Add Anthropic API key in your `.env` file. Place the textbook PDF in `data/`.

## Usage

```bash
# 1. Build the vector store (run once, or whenever chunking logic changes)
python ingest.py --pdf data/textbook.pdf

# 2. Launch the app
streamlit run app.py
```

## Architecture

```
data/textbook.pdf
      |
      v
ingest.py: pdf_to_markdown -> chunk_markdown -> embed_chunks -> store_in_chroma
      |
      v
chroma_db/  (persistent vector store)
      |
      v
rag.py: retrieve() -> _format_context() -> generate_practice_questions() / answer_conceptual_question()
      |
      v
app.py (Streamlit UI)
```

## Architecture decisions

- **Markdown conversion:** 

      -     Initial research pointed to PyMuPDF as a popular choice for PDF-to-Markdown 
      conversion. However, it struggled to convert math equations efficiently, and the
      resulting output was polluted with unknown symbols.

      -     The second candidate was Nougat, which seemed promising at first — it was
      purpose-built as a PDF parser that understands LaTeX math and tables. This too
      ran into issues: it has not been actively maintained by Meta since August 2023
      (as of June 2026), and the library now suffers from significant dependency rot,
       breaking across several transitive dependencies (`albumentations`, `pypdfium2`, 
       `transformers`) on any modern environment.
      
      -     This led to a third candidate, Marker, which handles LaTeX equations well.
       This came with one final hurdle: since development was done on an Apple Silicon 
       MacBook, a known regression in Marker 1.9.0+ causes its table recognition model 
       to be incompatible with the MPS backend, silently falling back to CPU and 
       resulting in significantly longer runtimes. Pinning Marker to version 1.8.0 
       resolved this, producing markdown conversion that is both accurate and rich in 
       metadata, enabling easier downstream post-processing.

- **Chunking strategy:**

      -     Fixed-size chunking was ruled out early, since splitting at character boundaries
      cuts proofs in half and orphans definitions from their theorems. Instead, the
      text is sliced at numbered subsection headings, with labeled blocks (Definition,
      Theorem, Lemma, Example, etc.) additionally extracted as standalone chunks. A
      soft cap is implemented instead of hard cap to prevent splitting mid-section and
      preserve structural integrity.

- **Embedding model:**

      -     A local model was preferred to avoid per-query API costs. `bge-base-en-v1.5`
      was the initial candidate, but profiling the actual chunks revealed an average of
      ~730 tokens and a max of ~4,900 — well above its 512-token limit, meaning most
      chunks would be silently truncated. `nomic-ai/nomic-embed-text-v1` was chosen
      instead for its 8,192-token context window. `max_seq_length` is capped at 2,048
      tokens in practice due to memory constraints on Apple Silicon, 46 of 424 chunks
      exceed this and have their tails truncated. Hard-splitting oversized chunks at
      paragraph boundaries would fix this. 17 of the 46 truncated chunks are labeled
      blocks (Theorems, Lemmas, Definitions, Examples), meaning primary retrieval
      targets are affected — not just surrounding prose. The fix is deferred in favor
      of validating retrieval quality first, but hard-splitting is a likely next step 
      in the future.
- **Retrieval k / thresholding:** [how many chunks, what happens on weak matches]
- **Prompt injection mitigation:** [how context is delimited / treated as data]
- **What you'd improve with more time:** [reranking, eval harness, hybrid search, etc.]

## Security note

This app retrieves and injects textbook content directly into LLM prompts.
Retrieved chunks are treated as data, not instructions --- prompts explicitly
tell the model to ignore any instruction-like text found within retrieved
context. See rag.py `_format_context()`.
