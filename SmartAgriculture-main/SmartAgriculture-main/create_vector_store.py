# create_vector_store.py
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from knowledge_base import KNOWLEDGE_BASE

print(f"Knowledge-base records found: {len(KNOWLEDGE_BASE)}")

# Validate IDs and required fields before building the index.
ids = [item["id"] for item in KNOWLEDGE_BASE]
if len(ids) != len(set(ids)):
    raise ValueError("Duplicate IDs found in KNOWLEDGE_BASE.")

required = {"id", "text", "source", "url", "topic"}
for item in KNOWLEDGE_BASE:
    missing = required - item.keys()
    if missing:
        raise ValueError(f"Record {item.get('id')} is missing fields: {missing}")
    if not item["text"].strip():
        raise ValueError(f"Record {item['id']} has empty text.")

documents = []
for item in KNOWLEDGE_BASE:
    content = (
        f"Topic: {item['topic']}\n"
        f"Information: {item['text']}\n"
        f"Source: {item['source']}\n"
        f"Reference: {item['url']}"
    )
    documents.append(
        Document(
            page_content=content,
            metadata={
                "id": item["id"],
                "topic": item["topic"],
                "source": item["source"],
                "url": item["url"],
            },
        )
    )

print(f"Documents prepared: {len(documents)}")
print("Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Building FAISS index...")
vectorstore = FAISS.from_documents(documents, embedding_model)

Path("vector_store").mkdir(exist_ok=True)
vectorstore.save_local("vector_store")

print(f"FAISS vectors saved: {vectorstore.index.ntotal}")
if vectorstore.index.ntotal != len(KNOWLEDGE_BASE):
    raise RuntimeError("FAISS vector count does not match knowledge-base record count.")

print("Vector store created successfully.")
