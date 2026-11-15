import os
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# -------------------------
# Configuration
# -------------------------
st.set_page_config(page_title="Smart Agriculture RAG Bot", page_icon="🌾", layout="wide")
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# Set this in .env to a model currently available in your OpenRouter account.
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b")
VECTOR_STORE_PATH = "vector_store"
TOP_K = 5

st.title("🌾 Smart Agriculture RAG Bot")
st.caption("Ask educational questions about crops, soil, irrigation, farm practices and post-harvest processes.")

if not OPENROUTER_API_KEY:
    st.error("OPENROUTER_API_KEY is missing. Add it to your .env file.")
    st.stop()

# -------------------------
# Load vector store
# -------------------------
@st.cache_resource
def load_vector_store():
    if not os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss")):
        raise FileNotFoundError(
            "FAISS index not found. Run: python create_vector_store.py"
        )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )

try:
    vectorstore = load_vector_store()
except Exception as exc:
    st.error(f"Could not load the FAISS index: {exc}")
    st.stop()

st.sidebar.subheader("Knowledge base status")
st.sidebar.write(f"Indexed documents: **{vectorstore.index.ntotal}**")
st.sidebar.write(f"LLM model: `{OPENROUTER_MODEL}`")
st.sidebar.caption("After changing knowledge_base.py, rebuild the vector store.")

# -------------------------
# Retrieve relevant context
# -------------------------
def retrieve_documents(question: str):
    # MMR balances relevance with diversity, helping avoid near-duplicate chunks.
    return vectorstore.max_marginal_relevance_search(
        question,
        k=min(TOP_K, vectorstore.index.ntotal),
        fetch_k=min(15, vectorstore.index.ntotal),
        lambda_mult=0.7,
    )

# -------------------------
# Generate answer with OpenRouter
# -------------------------
def generate_answer(question: str, docs):
    context_parts = []
    for i, doc in enumerate(docs, start=1):
        context_parts.append(
            f"[Document {i}]\n"
            f"Topic: {doc.metadata.get('topic', 'unknown')}\n"
            f"Content: {doc.page_content}\n"
            f"Source: {doc.metadata.get('source', 'Not specified')}\n"
            f"Reference: {doc.metadata.get('url', 'Not specified')}"
        )
    context = "\n\n".join(context_parts) if context_parts else "No relevant documents were retrieved."

    prompt = f"""
You are a helpful Smart Agriculture educational assistant.

Answer the user's agriculture-related question clearly and practically.
Use the retrieved knowledge-base documents when they are relevant.

Important rules:
1. First check whether the retrieved documents contain enough information.
2. If they do, base the answer on them and do not add unsupported claims.
3. If they do not contain enough information, you may still give a general educational
   answer using your general agricultural knowledge. Clearly add this note:
   "This part is general information and is not covered by the current knowledge base."
4. Never pretend that general model knowledge came from a retrieved source.
