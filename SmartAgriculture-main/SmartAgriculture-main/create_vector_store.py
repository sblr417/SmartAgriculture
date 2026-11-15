
import os

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

import knowledge_base

print("Knowledge base file location:")
print(knowledge_base.__file__)

print("Total documents loaded:", len(KNOWLEDGE_BASE))


# --------------------------------
# 1. LOAD KNOWLEDGE BASE
# --------------------------------

print("Loading agriculture knowledge base...")

documents = []

for item in KNOWLEDGE_BASE:

    content = f"""
{item['text']}

Source: {item['source']}
Reference: {item['url']}
Topic: {item['topic']}
"""

    document = Document(
        page_content=content,
        metadata={
            "id": item["id"],
            "topic": item["topic"],
            "source": item["source"],
            "url": item["url"]
        }
    )

    documents.append(document)


print("Total knowledge documents:", len(documents))


# --------------------------------
# 2. VALIDATE KNOWLEDGE BASE
# --------------------------------

if not documents:
    raise ValueError("Knowledge base is empty!")

if len(documents) != len(KNOWLEDGE_BASE):
    raise ValueError("Document count mismatch!")

definition_found = any(
    "Agriculture is the practice" in doc.page_content
    for doc in documents
)

if not definition_found:
    raise ValueError(
        "Agriculture definition is missing from knowledge_base.py!"
    )

print("Agriculture definition found in source data.")


# --------------------------------
# 3. LOAD EMBEDDING MODEL
# --------------------------------

print("Loading embedding model...")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully!")


# --------------------------------
# 4. CREATE FAISS VECTOR STORE
# --------------------------------

print("Generating embeddings and creating FAISS index...")

vectorstore = FAISS.from_documents(
    documents=documents,
    embedding=embedding_model
)

print("FAISS vector store created successfully!")


# --------------------------------
# 5. SAVE VECTOR STORE
# --------------------------------

base_dir = os.path.dirname(os.path.abspath(__file__))

vector_path = os.path.join(
    base_dir,
    "vector_store"
)

os.makedirs(vector_path, exist_ok=True)

vectorstore.save_local(vector_path)

print("FAISS index saved successfully!")
print("Saved location:", vector_path)


# --------------------------------
# 6. VERIFY SAVED FILES
# --------------------------------

faiss_file = os.path.join(
    vector_path,
    "index.faiss"
)

pkl_file = os.path.join(
    vector_path,
    "index.pkl"
)

if not os.path.isfile(faiss_file):
    raise FileNotFoundError("index.faiss was not created!")

if not os.path.isfile(pkl_file):
    raise FileNotFoundError("index.pkl was not created!")

print("Both FAISS files are available.")


# --------------------------------
# 7. RELOAD AND VERIFY INDEX
# --------------------------------

print("Reloading saved FAISS index...")

loaded_store = FAISS.load_local(
    vector_path,
    embedding_model,
    allow_dangerous_deserialization=True
)

indexed_documents = list(
    loaded_store.docstore._dict.values()
)

print("Verified indexed documents:", len(indexed_documents))


# --------------------------------
# 8. VERIFY AGRICULTURE DEFINITION
# --------------------------------

definition_present = any(
    "Agriculture is the practice" in doc.page_content
    for doc in indexed_documents
)

print(
    "Agriculture definition present:",
    definition_present
)


# --------------------------------
# 9. TEST SIMILARITY SEARCH
# --------------------------------

print("\nTesting agriculture definition retrieval...")

results = loaded_store.similarity_search(
    "What is agriculture?",
    k=5
)

for index, doc in enumerate(results, start=1):

    print(f"\nDocument {index}")
    print("Topic:", doc.metadata.get("topic"))
    print("Content:", doc.page_content[:300])


print("\nFAISS VECTOR STORE GENERATION COMPLETED!")
