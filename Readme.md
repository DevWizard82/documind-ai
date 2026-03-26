# 🧠 DocuMind AI

> Upload any PDF and chat with it using AI. Get instant answers with source citations and key concept breakdowns — powered by Google Gemini and FAISS vector search.<br/>
[![Live Demo](https://img.shields.io/badge/Live_Demo-View_Site-deb887?style=for-the-badge)](https://x2yvs58sxktxjdtdr3gx8l.streamlit.app/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://documind-ai.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-4285F4?style=flat&logo=google&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## ✨ Features

- **PDF upload & indexing** — drag and drop any PDF, indexed in ~15 seconds
- **RAG-powered answers** — retrieves the most relevant chunks before answering, no hallucination
- **Source citations** — every answer links back to the exact chunks it used
- **Key concepts card** — toggle to extract terms, definitions and difficulty level for any answer
- **Session memory** — full chat history with concepts persisted across the conversation
- **Clean dark UI** — minimal Streamlit interface with gold accent theme

---

## 🧠 How It Works

```
PDF upload
    │
    ▼
Text extraction (PyPDF2)
    │
    ▼
Chunking (500 words, 50-word overlap)
    │
    ▼
Embedding (Gemini Embedding 2 Preview → 3072-dim vectors)
    │
    ▼
FAISS index (stored locally)
    │
    ▼
User question → embedded → top-5 chunks retrieved
    │
    ▼
Gemini 2.5 Flash → grounded answer + key concepts
```

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| LLM | Gemini 2.5 Flash |
| Embeddings | Gemini Embedding 2 Preview (3072-dim) |
| Vector Store | FAISS (local, CPU) |
| PDF Parsing | PyPDF2 |
| UI + Deployment | Streamlit Cloud |
| API Client | google-genai SDK |

---

## ⚙️ Run Locally

**Prerequisites:** Python 3.10+, a free [Google AI Studio](https://aistudio.google.com) API key

```bash
# 1. Clone
git clone https://github.com/DevWizard82/documind-ai.git
cd documind-ai

# 2. Virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API key
echo GOOGLE_API_KEY=your_key_here > .env

# 5. Run
streamlit run app.py
```
