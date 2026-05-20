import streamlit as st
from rag import (
    extract_text_from_pdf,
    chunk_text,
    build_vector_store,
    retrieve_relevant_chunks,
    chat_with_groq
)

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 RAG Chatbot")
st.caption("Chat freely or upload a PDF to ask questions about it!")

# ── Session State ────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# ── Sidebar — PDF Upload ─────────────────────────────────────
with st.sidebar:
    st.header("📄 Upload a PDF (optional)")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_file:
        if uploaded_file.name != st.session_state.pdf_name:
            with st.spinner("Processing PDF..."):
                text = extract_text_from_pdf(uploaded_file)
                chunks = chunk_text(text)
                index, _ = build_vector_store(chunks)
                st.session_state.vector_store = index
                st.session_state.chunks = chunks
                st.session_state.pdf_name = uploaded_file.name
            st.success(f"✅ '{uploaded_file.name}' loaded!")
            st.info(f"📚 {len(chunks)} chunks indexed")

    if st.session_state.pdf_name:
        st.divider()
        st.caption(f"Active PDF: **{st.session_state.pdf_name}**")
        if st.button("🗑️ Remove PDF"):
            st.session_state.vector_store = None
            st.session_state.chunks = None
            st.session_state.pdf_name = None
            st.rerun()

    st.divider()
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# ── Chat History Display ─────────────────────────────────────
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat Input ───────────────────────────────────────────────
if prompt := st.chat_input("Ask me anything..."):

    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve context if PDF is loaded
    context_chunks = None
    if st.session_state.vector_store is not None:
        context_chunks = retrieve_relevant_chunks(
            prompt,
            st.session_state.vector_store,
            st.session_state.chunks
        )

    # Get response from Groq
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_with_groq(
                prompt,
                st.session_state.chat_history,
                context_chunks
            )
        st.markdown(response)

    # Save to history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.session_state.chat_history.append({"role": "assistant", "content": response})