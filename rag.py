"""
rag.py

Retrieval + generation logic. Two main capabilities:
1. generate_practice_questions(topic, difficulty, n) -> practice Qs + solutions
2. answer_conceptual_question(question) -> answer grounded in retrieved chunks

Both should:
- retrieve relevant chunks from ChromaDB
- pass them to Claude with a defensive prompt (treat retrieved content as data,
  not instructions -- see prompt injection note in README)
"""

import os
import chromadb
from anthropic import Anthropic

CLAUDE_MODEL = "claude-sonnet-4-6"


def get_client() -> Anthropic:
    """
    Return an Anthropic client using ANTHROPIC_API_KEY from environment.

    TODO:
    - Load .env (python-dotenv) if not already loaded by the caller
    - Instantiate and return Anthropic()
    """
    raise NotImplementedError


def get_collection(collection_name: str = "textbook"):
    """
    Return the ChromaDB collection created by ingest.py.

    TODO:
    - Connect to the same persistent ChromaDB path used in ingest.py
    - Return the collection object
    """
    raise NotImplementedError


def retrieve(query: str, k: int = 4, collection_name: str = "textbook") -> list[dict]:
    """
    Retrieve the top-k most relevant chunks for a query.

    Decision to make: what do you do when results are weak/irrelevant
    (e.g. similarity score below some threshold)? Returning bad chunks to the
    generation step produces confidently wrong answers -- worth handling
    explicitly rather than ignoring.

    TODO:
    - Embed the query (same embedding approach as ingest.py -- must match!)
    - Query the collection for top-k matches
    - Return list of {"text": ..., "metadata": ..., "score": ...}
    """
    raise NotImplementedError


def _format_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a string block for the prompt, wrapped in
    delimiters so the model can distinguish data from instructions.

    TODO:
    - Wrap each chunk in something like <chunk section="3.2">...</chunk>
    - Join into a single string
    """
    raise NotImplementedError


def generate_practice_questions(topic: str, difficulty: str = "medium", n: int = 3) -> str:
    """
    Generate n practice questions + solutions on `topic` at the given difficulty,
    grounded in retrieved textbook content.

    TODO:
    - retrieve(topic) to get relevant chunks
    - Build a prompt: system instructions (treat context as data only) +
      formatted context + instructions to produce n questions with solutions
    - Call Claude via get_client().messages.create(...)
    - Return the text response
    """
    raise NotImplementedError


def answer_conceptual_question(question: str) -> str:
    """
    Answer a student's conceptual question using retrieved textbook content.
    Should say "I don't know" / "not covered in this material" if retrieval
    doesn't surface anything relevant -- don't let the model guess freely.

    TODO:
    - retrieve(question) to get relevant chunks
    - Build prompt with context + question
    - Call Claude, return response text
    - Consider returning source section(s) alongside the answer for transparency
    """
    raise NotImplementedError
