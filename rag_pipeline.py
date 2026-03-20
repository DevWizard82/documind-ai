import os
import pickle
import numpy as np
from dotenv import load_dotenv
from google import genai
import PyPDF2
import faiss

load_dotenv()

def get_client():
    try:
        import streamlit as st
        api_key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        api_key = os.getenv("GOOGLE_API_KEY")
    return genai.Client(api_key=api_key)


client = get_client()

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
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
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
def embed_texts(texts: list[str]) -> np.ndarray:
    embeddings = []
    for text in texts:
        response = client.models.embed_content(
            model="gemini-embedding-2-preview",
            contents=text,
        )
        embeddings.append(response.embeddings[0].values)
    return np.array(embeddings, dtype="float32")

# ── 4. BUILD FAISS INDEX ──────────────────────────────────────────────────────
def build_index(chunks: list[str]) -> tuple:
    print(f"Embedding {len(chunks)} chunks...")
    embeddings = embed_texts(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"Index built with {index.ntotal} vectors of dimension {dim}")
    return index, chunks

# ── 5. RETRIEVAL ──────────────────────────────────────────────────────────────
def retrieve(query: str, index, chunks: list[str], top_k: int = 5) -> list[str]:
    query_embedding = embed_texts([query])
    _, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]

# ── 6. ANSWER GENERATION ──────────────────────────────────────────────────────
def answer_question(query: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    prompt = f"""You are a helpful assistant answering questions based ONLY on the provided document context.
If the answer is not in the context, say "I couldn't find this information in the document."

Context:
{context}

Question: {query}

Answer:"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text

# ── 7. SAVE / LOAD INDEX ──────────────────────────────────────────────────────
def save_index(index, chunks: list[str], path: str = "index_store"):
    os.makedirs(path, exist_ok=True)
    faiss.write_index(index, f"{path}/index.faiss")
    with open(f"{path}/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print(f"Index saved to {path}/")

def load_index(path: str = "index_store") -> tuple:
    index = faiss.read_index(f"{path}/index.faiss")
    with open(f"{path}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    print(f"Index loaded — {index.ntotal} vectors")
    return index, chunks


def generate_diagram(query: str, answer: str, context_chunks: list[str]) -> str | None:
    context = "\n\n---\n\n".join(context_chunks)
    prompt = f"""Based on this question and answer about a document, generate a Mermaid diagram that visually explains the concept.

Question: {query}
Answer: {answer}
Document context: {context[:1500]}

Rules:
- Output ONLY valid Mermaid syntax, nothing else. No markdown fences, no explanation.
- Choose the most appropriate diagram type:
  * flowchart TD  → for processes, steps, algorithms
  * graph LR      → for relationships, comparisons
  * mindmap       → for concepts with sub-topics
- Keep it simple: max 8 nodes
- Use short labels (3-5 words max per node)
- If the content is purely factual with no clear structure to diagram, respond with exactly: SKIP

Example output for a process:
flowchart TD
    A[Start] --> B[Step one]
    B --> C[Step two]
    C --> D[End]"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    result = response.text.strip()
    if result == "SKIP" or len(result) < 10:
        return None
    # Strip accidental markdown fences if model adds them
    result = result.replace("```mermaid", "").replace("```", "").strip()
    return result


def generate_concept_card(query: str, answer: str) -> dict:
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
        model="gemini-2.5-flash",
        contents=prompt
    )
    import json, re
    text = response.text.strip()
    text = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        return None