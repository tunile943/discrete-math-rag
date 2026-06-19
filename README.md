# Discrete Structures RAG

Generate practice questions and answer conceptual questions grounded in a Discrete Structures textbook, using a retrieval-augmented generation pipeline.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Add Anthropic API key in your `.env` file. Place the textbook PDF in `data/`.

## Usage

```bash
# 1. Build the vector store (run once, or whenever chunking logic changes)
python ingest.py --pdf data/your_textbook.pdf # Replace `data/your_textbook.pdf` with your actual file name.

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
