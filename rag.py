import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

# Load the embedding model (runs locally, no API needed)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── PDF Processing ──────────────────────────────────────────
def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

# ── FAISS Vector Store ──────────────────────────────────────
def build_vector_store(chunks):
    """Embed chunks and store in FAISS index"""
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings

def retrieve_relevant_chunks(query, index, chunks, top_k=3):
    """Find the most relevant chunks for a query"""
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]

# ── Groq Chat ───────────────────────────────────────────────
def chat_with_groq(user_message, chat_history, context_chunks=None):
    """Send message to Groq with optional RAG context"""
    
    system_prompt = "You are a helpful, friendly AI assistant."
    
    if context_chunks:
        context = "\n\n".join(context_chunks)
        system_prompt = f"""You are a helpful AI assistant. 
Answer the user's question based on the context below.
If the answer isn't in the context, say so honestly.

CONTEXT:
{context}"""

    messages = [{"role": "system", "content": system_prompt}]
    
    # Add chat history
    for msg in chat_history:
        messages.append(msg)
    
    # Add current message
    messages.append({"role": "user", "content": user_message})
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.7,
    )
    
    return response.choices[0].message.content
