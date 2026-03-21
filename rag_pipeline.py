import os
import pickle
import json
import re
import numpy as np
import faiss
import PyPDF2
from dotenv import load_dotenv
from google import genai
from groq import Groq

load_dotenv()

# ── Lazy clients ──────────────────────────────────────────────────────────────
_gemini_client = None
_groq_client   = None

def get_gemini_client():
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client
    try:
        import streamlit as st
        api_key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found")
    _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client

def get_groq_client():
    global _groq_client
    if _groq_client is not None:
        return _groq_client
    try:
        import streamlit as st
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found")
    _groq_client = Groq(api_key=api_key)
    return _groq_client

def retrieve_per_document(query: str, index, chunks: list, top_k_per_doc: int = 3) -> list:
    """Retrieves top_k chunks from EACH document — guarantees all docs are represented."""
    # Group chunk indices by source document
    from collections import defaultdict
    doc_indices = defaultdict(list)
    for i, chunk in enumerate(chunks):
        doc_indices[chunk["source"]].append(i)

    # Embed the query once
    query_embedding = embed_texts([query])

    results = []
    for source, indices in doc_indices.items():
        # Build a mini index for just this document's chunks
        doc_chunks     = [chunks[i] for i in indices]
        doc_embeddings = np.array(
            [embed_texts([c["text"]])[0] for c in doc_chunks],
            dtype="float32"
        )
        mini_index = faiss.IndexFlatL2(doc_embeddings.shape[1])
        mini_index.add(doc_embeddings)

        # Retrieve top_k from this document
        k          = min(top_k_per_doc, len(doc_chunks))
        _, mini_idx = mini_index.search(query_embedding, k)
        for idx in mini_idx[0]:
            if idx < len(doc_chunks):
                results.append(doc_chunks[idx])

    return results

# ── 1. PDF TEXT EXTRACTION — with page numbers ────────────────────────────────
def extract_text_from_pdf(pdf_path: str, filename: str = "") -> list:
    """Returns list of dicts: {text, page, source}"""
    pages = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                pages.append({
                    "text":   page_text,
                    "page":   i + 1,           # 1-indexed
                    "source": filename or os.path.basename(pdf_path),
                })
    return pages


# ── 2. TEXT CHUNKER — preserves page + source metadata ───────────────────────
def chunk_pages(pages: list, chunk_size: int = 500, overlap: int = 50) -> list:
    """Returns list of dicts: {text, page, source}"""
    chunks = []
    for page_data in pages:
        words  = page_data["text"].split()
        start  = 0
        while start < len(words):
            end   = start + chunk_size
            chunk = " ".join(words[start:end])
            chunks.append({
                "text":   chunk,
                "page":   page_data["page"],
                "source": page_data["source"],
            })
            start += chunk_size - overlap
    return chunks


# ── 3. EMBEDDING ──────────────────────────────────────────────────────────────
def embed_texts(texts: list) -> np.ndarray:
    client     = get_gemini_client()
    embeddings = []
    for text in texts:
        response = client.models.embed_content(
            model="models/gemini-embedding-2-preview",
            contents=text,
        )
        embeddings.append(response.embeddings[0].values)
    return np.array(embeddings, dtype="float32")


# ── 4. BUILD FAISS INDEX ──────────────────────────────────────────────────────
def build_index(chunks: list) -> tuple:
    """chunks: list of {text, page, source} dicts"""
    texts      = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    dim        = embeddings.shape[1]
    index      = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, chunks


# ── 5. RETRIEVAL ──────────────────────────────────────────────────────────────
def retrieve(query: str, index, chunks: list, top_k: int = 5) -> list:
    """Returns top-k chunk dicts with text, page, source"""
    query_embedding = embed_texts([query])
    _, indices      = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]


# ── 6. ANSWER GENERATION ─────────────────────────────────────────────────────
def answer_question(query: str, context_chunks: list) -> str:
    client = get_groq_client()

    # Build context with page references
    context_parts = []
    for c in context_chunks:
        context_parts.append(
            f"[{c['source']} — Page {c['page']}]\n{c['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a helpful assistant answering questions based ONLY on the provided document context.

Instructions:
- Answer naturally and coherently — never list page by page unless explicitly asked
- If asked to summarize, write a flowing paragraph summary, not a page-by-page breakdown
- If asked about multiple documents, address each document by name
- If the answer is not in the context, say "I couldn't find this information in the document."
- Always be concise and direct

Context:
{context}

Question: {query}

Answer:"""

    response = get_groq_client().chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


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

    response = get_groq_client().chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.choices[0].message.content.strip()
    text = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        return None
