import os
import pickle
import json
import re
import numpy as np
import faiss
import PyPDF2
from dotenv import load_dotenv
from google import genai

load_dotenv()

# ── Lazy client ───────────────────────────────────────────────────────────────
_client = None

def get_client():
    global _client
    if _client is not None:
        return _client
    try:
        import streamlit as st
        api_key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in secrets or .env")
    _client = genai.Client(api_key=api_key)
    return _client


# ── 1. PDF TEXT EXTRACTION ────────────────────────────────────────────────────
def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


# ── 2. TEXT CHUNKER ───────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ── 3. EMBEDDING ──────────────────────────────────────────────────────────────
def embed_texts(texts: list) -> np.ndarray:
    client = get_client()
    embeddings = []
    for text in texts:
        response = client.models.embed_content(
            model="modelsgemini-embedding-2-preview",
            contents=text,
        )
        embeddings.append(response.embeddings[0].values)
    return np.array(embeddings, dtype="float32")


# ── 4. BUILD FAISS INDEX ──────────────────────────────────────────────────────
def build_index(chunks: list) -> tuple:
    embeddings = embed_texts(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, chunks


# ── 5. RETRIEVAL ──────────────────────────────────────────────────────────────
def retrieve(query: str, index, chunks: list, top_k: int = 5) -> list:
    query_embedding = embed_texts([query])
    _, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]


# ── 6. ANSWER GENERATION ─────────────────────────────────────────────────────
def answer_question(query: str, context_chunks: list) -> str:
    client = get_client()
    context = "\n\n---\n\n".join(context_chunks)
    prompt = f"""You are a helpful assistant answering questions based ONLY on the provided document context.
If the answer is not in the context, say "I couldn't find this information in the document."

Context:
{context}

Question: {query}

Answer:"""
    response = client.models.generate_content(
        model="gemini-embedding-2-preview",
        contents=prompt
    )
    return response.text


# ── 7. SAVE / LOAD INDEX ──────────────────────────────────────────────────────
def save_index(index, chunks: list, path: str = "index_store"):
    os.makedirs(path, exist_ok=True)
    faiss.write_index(index, f"{path}/index.faiss")
    with open(f"{path}/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)


def load_index(path: str = "index_store") -> tuple:
    index = faiss.read_index(f"{path}/index.faiss")
    with open(f"{path}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


# ── 8. KEY CONCEPTS CARD ──────────────────────────────────────────────────────
def generate_concept_card(query: str, answer: str) -> dict:
    client = get_client()
    prompt = f"""Extract key concepts from this Q&A and return ONLY a JSON object, no markdown, no explanation.

Question: {query}
Answer: {answer}

Return this exact structure:
{{
  "summary": "One sentence max, the core takeaway",
  "concepts": [
    {{"term": "Term 1", "definition": "Short definition, max 10 words"}},
    {{"term": "Term 2", "definition": "Short definition, max 10 words"}},
    {{"term": "Term 3", "definition": "Short definition, max 10 words"}}
  ],
  "difficulty": "beginner" or "intermediate" or "advanced"
}}

Extract 2-4 concepts maximum. Keep definitions under 10 words each."""

    response = client.models.generate_content(
        model="gemini-embedding-2-preview",
        contents=prompt
    )
    text = response.text.strip()
    text = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        return None
