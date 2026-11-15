
import os
import streamlit as st
import requests
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# -------------------------
# STREAMLIT CONFIGURATION
# -------------------------

st.set_page_config(
    page_title="Smart Agriculture RAG Bot",
    page_icon="🌾",
    layout="wide"
)


# -------------------------
# LOAD API KEY
# -------------------------

load_dotenv()

OPENROUTER_API_KEY = st.secrets.get(
    "OPENROUTER_API_KEY",
    os.getenv("OPENROUTER_API_KEY", "")
)

OPENROUTER_API_KEY = OPENROUTER_API_KEY.strip()

if not OPENROUTER_API_KEY:
    st.error("OpenRouter API key is missing!")
    st.stop()

# -------------------------
# STREAMLIT UI
# -------------------------

st.title("🌾 Smart Agriculture Crop Process Explainer Bot")

st.markdown(
    "AI-powered educational assistant for crop lifecycle "
    "and farming processes."
)


# -------------------------
# LOAD EMBEDDING MODEL
# -------------------------

@st.cache_resource
def load_embedding_model():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# -------------------------
# LOAD FAISS VECTOR STORE
# -------------------------

@st.cache_resource
def load_vector_store():

    base_dir = os.path.dirname(os.path.abspath(__file__))

    vector_path = os.path.join(
        base_dir,
        "vector_store"
    )

    faiss_file = os.path.join(
        vector_path,
        "index.faiss"
    )

    pkl_file = os.path.join(
        vector_path,
        "index.pkl"
    )

    # Check vector_store folder

    if not os.path.isdir(vector_path):
        st.error(
            "Vector store folder is missing. "
            "Please upload the vector_store folder to GitHub."
        )
        st.stop()

    # Check FAISS index file

    if not os.path.isfile(faiss_file):
        st.error("index.faiss file is missing inside vector_store.")
        st.stop()

    # Check pickle file

    if not os.path.isfile(pkl_file):
        st.error("index.pkl file is missing inside vector_store.")
        st.stop()

    try:

        embedding_model = load_embedding_model()

        vectorstore = FAISS.load_local(
            vector_path,
            embedding_model,
            allow_dangerous_deserialization=True
        )

        return vectorstore

    except Exception as e:

        st.error(f"FAISS loading failed: {str(e)}")
        st.stop()


vectorstore = load_vector_store()


# -------------------------
# SAFETY FILTER
# -------------------------

def is_restricted(query):

    restricted_terms = [
        "fertilizer",
        "chemical dosage",
        "pesticide amount",
        "yield prediction",
        "how much urea",
        "profit estimate"
    ]

    return any(
        term in query.lower()
        for term in restricted_terms
    )


# -------------------------
# OPENROUTER LLM FUNCTION
# -------------------------

def generate_answer(query, context):

    headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://smartagriculture.streamlit.app",
    "X-OpenRouter-Title": "Smart Agriculture RAG Bot"
    }

    prompt = f"""
    You are an agriculture education assistant.

    Only provide informational explanations.

    Do NOT provide:
    - Fertilizer recommendations
    - Chemical dosage
    - Pesticide quantity
    - Yield prediction
    - Profit estimation

    Use the provided context to answer the question.

    If the context does not contain the answer,
    clearly mention that the information is not available
    in the knowledge base.

    Context:
    {context}

    Question:
    {query}

    Provide a simple explanation suitable for farmers.
    """

    data = {
        "model": "openrouter/free",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.3
    }

    try:

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )

        if response.status_code != 200:

            return (
                f"OpenRouter API Error "
                f"({response.status_code}): {response.text}"
            )

        result = response.json()

        if "choices" not in result or not result["choices"]:

            return "Unexpected response from OpenRouter API."

        return result["choices"][0]["message"]["content"]

    except requests.exceptions.Timeout:

        return "The API request timed out. Please try again."

    except requests.exceptions.RequestException as e:

        return f"Network error: {str(e)}"

    except Exception as e:

        return f"System error: {str(e)}"


# -------------------------
# QUERY INTERFACE
# -------------------------

st.markdown("---")

user_query = st.text_input(
    "Ask your agriculture-related question:",
    placeholder="Example: Explain the rice cultivation process"
)


# -------------------------
# PROCESS USER QUERY
# -------------------------

if user_query:

    if is_restricted(user_query):

        st.warning(
            "⚠️ This system provides educational explanations only."
        )

    else:

        try:

            with st.spinner("🔎 Searching knowledge base..."):

                docs = vectorstore.similarity_search(
                    user_query,
                        k=5
                )
                    
                st.write("### DEBUG: Retrieved Documents")
                    
                for i, doc in enumerate(docs):
                    st.write(f"Document {i + 1}")
                    st.write(doc.page_content)
                    st.write("Metadata:", doc.metadata)
                    
                context = "\n\n".join(
                        [doc.page_content for doc in docs]
                )

            if not docs:

                st.warning(
                    "No relevant information found in the knowledge base."
                )

            else:

                with st.spinner("🤖 Generating response..."):

                    answer = generate_answer(
                        user_query,
                        context
                    )

                st.success("✅ Answer")

                st.write(answer)

                # -------------------------
                # DISPLAY SOURCES
                # -------------------------

                st.markdown("---")
                st.markdown("### 📖 Sources")

                for doc in docs:

                    source = doc.metadata.get(
                        "source",
                        "Knowledge Base"
                    )

                    st.write(f"- {source}")

        except Exception as e:

            st.error(
                f"Error while processing your question: {str(e)}"
            )
