import os

import requests
import streamlit as st

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")


def call_ollama(prompt: str, model: str = DEFAULT_MODEL, temperature: float = 0.3) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 400,
        },
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=180)
    response.raise_for_status()
    data = response.json()
    return data.get("response", "").strip()


def summarize_notes(notes: str) -> str:
    prompt = f"""You are a helpful student study assistant.

Read the notes below and produce a clear, student-friendly summary.
Keep it simple, accurate, and focused on the main ideas.

Notes:
{notes}

Return:
1. A short summary in 4-6 sentences.
2. A heading called 'Key ideas:' followed by 3-5 bullet points.
3. A heading called 'Important terms:' followed by 3-5 important terms.
"""
    return call_ollama(prompt, temperature=0.2)


def key_points(notes: str) -> str:
    prompt = f"""List the most important facts, ideas, and concepts from these notes.
Focus on what a student should remember for an exam or revision.

Notes:
{notes}

Output only bullet points.
"""
    return call_ollama(prompt, temperature=0.2)


def revision_questions(notes: str) -> str:
    prompt = f"""Create 5 short revision questions based on the notes below.
Each question should test understanding, not just memory.
Then add a short answer key.

Notes:
{notes}
"""
    return call_ollama(prompt, temperature=0.4)


def explain_concept(notes: str, concept: str) -> str:
    prompt = f"""The student is learning about the concept '{concept}'.
Use the notes below to explain this concept in a clear, beginner-friendly way.
Include:
- a simple explanation
- a real-world example
- why it matters
- 2 short questions the student can ask themselves

Notes:
{notes}
"""
    return call_ollama(prompt, temperature=0.3)


st.set_page_config(page_title="LazyNotes AI", page_icon="📝", layout="wide")
st.title("LazyNotes AI")
st.caption("Local study notes helper powered by Ollama on your computer.")

with st.sidebar:
    st.header("Settings")
    model_name = st.text_input("Ollama model", value=DEFAULT_MODEL, help="Use a model already installed in Ollama.")
    st.info("Recommended: llama3.2:3b")

notes = st.text_area(
    "Paste your notes here",
    height=350,
    placeholder="Type or paste your class notes here...",
)

if not notes.strip():
    st.warning("Add your notes first, then choose an action.")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Summarize") and notes.strip():
        with st.spinner("Generating summary..."):
            result = summarize_notes(notes)
        st.subheader("Summary")
        st.write(result)

with col2:
    if st.button("Key points") and notes.strip():
        with st.spinner("Finding the important ideas..."):
            result = key_points(notes)
        st.subheader("Key points")
        st.write(result)

with col3:
    if st.button("Revision questions") and notes.strip():
        with st.spinner("Creating revision questions..."):
            result = revision_questions(notes)
        st.subheader("Revision questions")
        st.write(result)

st.markdown("---")

concept = st.text_input("Explain a concept from your notes", placeholder="Example: photosynthesis")

if st.button("Explain concept") and notes.strip() and concept.strip():
    with st.spinner("Explaining the concept..."):
        result = explain_concept(notes, concept)
    st.subheader(f"Understanding: {concept}")
    st.write(result)

st.caption("Tip: If model generation fails, confirm Ollama is running and the selected model is installed on your machine.")

