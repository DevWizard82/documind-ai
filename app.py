import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from rag_pipeline import (
    extract_text_from_pdf,
    chunk_text,
    build_index,
    retrieve,
    answer_question,
    save_index,
    generate_concept_card,
)

load_dotenv()

# ── SVG Icons ─────────────────────────────────────────────────────────────────
def icon(svg_body, size=18, color="#C9A96E"):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}"'
            f' viewBox="0 0 24 24" fill="none" stroke="{color}"'
            f' stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"'
            f' style="vertical-align:middle;margin-right:6px">{svg_body}</svg>')

ICO_DOC    = '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>'
ICO_CHAT   = '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>'
ICO_ARROW  = '<line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/>'
ICO_LOGO   = '<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>'
ICO_INFO   = '<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>'

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
div[data-testid="stFileUploader"] {
    background: rgba(201,169,110,0.06);
    border: 1.5px dashed rgba(201,169,110,0.3);
    border-radius: 12px;
    padding: 1rem;
}
div[data-testid="stFileUploader"]:hover {
    border-color: #C9A96E;
    background: rgba(201,169,110,0.10);
}
div[data-testid="stFileUploader"] label,
div[data-testid="stFileUploader"] span { color: #C9A96E !important; }
div[data-testid="stFileUploader"] small { opacity: .55; }
.step { display:flex; align-items:flex-start; gap:12px; margin:10px 0; }
.step-num {
    background: #C9A96E; color: #fff; border-radius: 50%;
    width:22px; height:22px; display:flex; align-items:center;
    justify-content:center; font-size:.75rem; font-weight:700;
    flex-shrink:0; margin-top:2px;
}
.logo-wrap { display:flex; align-items:center; gap:10px; padding:4px 0 12px; }
.logo-text  { font-size:1.4rem; font-weight:700; letter-spacing:-0.3px; }
.logo-text span { color: #C9A96E; }
.stat-badge {
    display:inline-block;
    background:rgba(201,169,110,.12);
    color:#C9A96E;
    border:1px solid rgba(201,169,110,.3);
    border-radius:20px;
    padding:2px 10px;
    font-size:.78rem; font-weight:600;
    margin:2px;
}
details summary { color:#C9A96E !important; font-size:.82rem; }
hr { border-color:rgba(201,169,110,.2) !important; }
div[data-testid="stChatInput"] textarea:focus { border-color:#C9A96E !important; }
.stButton > button:hover { border-color:#C9A96E !important; color:#C9A96E !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for key, val in [("index", None), ("chunks", None),
                 ("messages", []), ("doc_name", None)]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── Helper: render concepts card ─────────────────────────────────────────────
def render_concepts(card):
    difficulty_color = {
        "beginner":     "#4CAF50",
        "intermediate": "#C9A96E",
        "advanced":     "#E57373",
    }.get(card.get("difficulty", "intermediate"), "#C9A96E")

    concepts_html = "".join([
        f"""<div style="display:flex;gap:10px;align-items:flex-start;
                        margin:8px 0;padding:8px;
                        background:rgba(255,255,255,0.03);
                        border-radius:8px;
                        border-left:3px solid #C9A96E">
              <div style="min-width:120px;font-weight:600;
                          font-size:0.82rem;color:#C9A96E">{c['term']}</div>
              <div style="font-size:0.82rem;opacity:0.85">{c['definition']}</div>
            </div>"""
        for c in card.get("concepts", [])
    ])

    st.markdown(f"""
    <div style="background:rgba(201,169,110,0.06);
                border:1px solid rgba(201,169,110,0.2);
                border-radius:12px;padding:1rem;margin-top:0.5rem">
      <div style="display:flex;justify-content:space-between;
                  align-items:center;margin-bottom:10px">
        <span style="font-weight:600;font-size:0.88rem;color:#C9A96E">Key concepts</span>
        <span style="font-size:0.72rem;font-weight:600;
                     color:{difficulty_color};
                     border:1px solid {difficulty_color};
                     border-radius:20px;padding:1px 8px">
          {card.get('difficulty','').upper()}
        </span>
      </div>
      <div style="font-size:0.85rem;opacity:0.7;margin-bottom:10px;font-style:italic">
        {card.get('summary','')}
      </div>
      {concepts_html}
    </div>""", unsafe_allow_html=True)


# ── Helper: process uploaded PDF ─────────────────────────────────────────────
def process_upload(uploaded_file):
    with st.spinner("Indexing document — this takes ~15 seconds..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        text          = extract_text_from_pdf(tmp_path)
        chunks        = chunk_text(text)
        index, chunks = build_index(chunks)
        save_index(index, chunks)
        st.session_state.index    = index
        st.session_state.chunks   = chunks
        st.session_state.doc_name = uploaded_file.name
        st.session_state.messages = []
        os.unlink(tmp_path)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div class="logo-wrap">
      {icon(ICO_LOGO, 26, "#C9A96E")}
      <div class="logo-text">Docu<span>Mind</span> AI</div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Chat with your documents using AI")
    st.divider()

    if st.session_state.doc_name:
        st.markdown(f"{icon(ICO_DOC, 14)} **Active document**", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size:.82rem;opacity:.8;margin:4px 0 8px;word-break:break-word">
          {st.session_state.doc_name}
        </div>
        <span class="stat-badge">{len(st.session_state.chunks)} chunks</span>
        <span class="stat-badge">{len(st.session_state.messages)//2} Q&amp;As</span>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Clear chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        with col_b:
            if st.button("New doc", use_container_width=True, type="secondary"):
                for k in ["index", "chunks", "doc_name"]:
                    st.session_state[k] = None
                st.session_state.messages = []
                st.rerun()
    else:
        st.markdown(
            f"{icon(ICO_INFO, 14, '#888')}"
            "<span style='font-size:.82rem;opacity:.6'> No document loaded yet.<br>"
            "Upload one on the main page.</span>",
            unsafe_allow_html=True
        )

    st.divider()
    st.markdown(
        f"{icon(ICO_INFO, 13, '#888')}"
        "<span style='font-size:.75rem;color:#888'> Gemini 2.5 Flash + FAISS</span>",
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# LANDING PAGE — no document loaded
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.index:

    _, col, _ = st.columns([0.5, 3, 0.5])
    with col:
        st.markdown("<br>", unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload a PDF",
            type=["pdf"],
            label_visibility="visible",
            key="landing_uploader",
        )
        if uploaded:
            process_upload(uploaded)
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        left, right = st.columns(2, gap="large")

        with left:
            st.markdown("**How it works**")
            for num, step in enumerate([
                "Upload a PDF above",
                "Wait ~15 s for indexing",
                "Ask anything about it",
                "Get answers with source citations",
            ], 1):
                st.markdown(f"""
                <div class="step">
                  <div class="step-num">{num}</div>
                  <div style="font-size:.88rem;padding-top:3px">{step}</div>
                </div>""", unsafe_allow_html=True)

        with right:
            st.markdown("**Good questions to try**")
            for q in [
                "What is this document about?",
                "Summarize the key points",
                "Explain [topic] from the document",
                "What does the author conclude?",
            ]:
                st.markdown(
                    f"<div style='font-size:.85rem;color:#C9A96E;margin:8px 0'>"
                    f"{icon(ICO_ARROW, 13)} {q}</div>",
                    unsafe_allow_html=True
                )


# ══════════════════════════════════════════════════════════════════════════════
# CHAT PAGE — document loaded
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown(
        f"{icon(ICO_CHAT, 20)} **{st.session_state.doc_name}**",
        unsafe_allow_html=True
    )
    st.divider()

    # ── Render full chat history ──────────────────────────────────────────────
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg["role"] == "assistant":

                # Key concepts toggle
                st.markdown("<br>", unsafe_allow_html=True)
                show_concepts = st.toggle(
                    "Key concepts",
                    key=f"conc_{i}",
                    value=bool(msg.get("concepts"))
                )

                if show_concepts:
                    if msg.get("concepts"):
                        render_concepts(msg["concepts"])
                    else:
                        with st.spinner("Extracting concepts..."):
                            card = generate_concept_card(
                                msg.get("prompt", ""),
                                msg["content"]
                            )
                        if card:
                            st.session_state.messages[i]["concepts"] = card
                            render_concepts(card)
                        else:
                            st.caption("Could not extract concepts for this answer.")

                # Source chunks
                if msg.get("sources"):
                    with st.expander("View source chunks", expanded=False):
                        for j, chunk in enumerate(msg["sources"], 1):
                            st.markdown(f"**Chunk {j}**")
                            st.caption(chunk[:400] + "..." if len(chunk) > 400 else chunk)
                            if j < len(msg["sources"]):
                                st.divider()

    # ── Chat input ────────────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask anything about your document..."):

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                relevant_chunks = retrieve(
                    prompt,
                    st.session_state.index,
                    st.session_state.chunks,
                )
                answer = answer_question(prompt, relevant_chunks)
            st.markdown(answer)

        st.session_state.messages.append({
            "role":     "assistant",
            "content":  answer,
            "prompt":   prompt,
            "sources":  relevant_chunks,
            "concepts": None,
        })

        st.rerun()