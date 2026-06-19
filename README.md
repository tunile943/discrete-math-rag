# Discrete Structures RAG

Generate practice questions and answer conceptual questions grounded in a Discrete Math textbook, 
      using a retrieval-augmented generation pipeline.

### Built With

* [![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
* [![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
* [![Anthropic Claude](https://img.shields.io/badge/Anthropic%20Claude-191919?style=for-the-badge&logo=anthropic&logoColor=white)](https://www.anthropic.com/)
* [![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B35?style=for-the-badge&logoColor=white)](https://www.trychroma.com/)
* [![PyMuPDF](https://img.shields.io/badge/PyMuPDF-48A9A6?style=for-the-badge&logoColor=white)](https://pymupdf.readthedocs.io/)

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Add Anthropic API key in your `.env` file. Place the textbook PDF in `data/`.

## Usage

```bash
# 1. Build the vector store (run once, or whenever chunking logic changes)
python ingest.py --pdf data/textbook.pdf # Replace `data/textbook.pdf` with your actual file name.

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

- **Chunking strategy:** [section-based vs. fixed-size, and why]
- **Embedding model:** [which one, and why]
- **Retrieval k / thresholding:** [how many chunks, what happens on weak matches]
- **Prompt injection mitigation:** [how context is delimited / treated as data]
- **What you'd improve with more time:** [reranking, eval harness, hybrid search, etc.]

## Security note

This app retrieves and injects textbook content directly into LLM prompts.
Retrieved chunks are treated as data, not instructions --- prompts explicitly
tell the model to ignore any instruction-like text found within retrieved
context. See rag.py `_format_context()`.
