"""
app.py

Streamlit UI shell. This file is mostly working plumbing. Run with:

    streamlit run app.py
"""

import streamlit as st
from rag import generate_practice_questions, answer_conceptual_question

st.set_page_config(page_title="CS Textbook RAG", layout="centered")

st.title("CS Textbook Study Assistant")
st.caption("Practice questions and conceptual Q&A grounded in your textbook.")

mode = st.radio("What do you want to do?", ["Generate practice questions", "Ask a conceptual question"])

if mode == "Generate practice questions":
    topic = st.text_input("Topic or section (e.g. 'induction', 'pigeonhole principle', 'probability space')")
    difficulty = st.select_slider("Difficulty", options=["easy", "medium", "hard"], value="medium")
    n = st.slider("Number of questions", min_value=1, max_value=5, value=3)

    if st.button("Generate", type="primary") and topic:
        with st.spinner("Retrieving context and generating questions..."):
            try:
                result = generate_practice_questions(topic, difficulty=difficulty, n=n)
                st.markdown(result)
            except NotImplementedError:
                st.error("generate_practice_questions() isn't implemented yet -- fill in rag.py")
            except Exception as e:
                st.error(f"Something went wrong: {e}")

else:
    question = st.text_area("Your question")

    if st.button("Ask", type="primary") and question:
        with st.spinner("Retrieving context and answering..."):
            try:
                result = answer_conceptual_question(question)
                st.markdown(result)
            except NotImplementedError:
                st.error("answer_conceptual_question() isn't implemented yet -- fill in rag.py")
            except Exception as e:
                st.error(f"Something went wrong: {e}")

st.divider()
st.caption("Run `python ingest.py --pdf data/textbook.pdf` first to build the vector store.")
