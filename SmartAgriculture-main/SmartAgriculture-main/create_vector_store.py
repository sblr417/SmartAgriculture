
import os

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from knowledge_base import KNOWLEDGE_BASE

documents = []

for item in KNOWLEDGE_BASE:
    content = f"""
{item['text']}

Source: {item['source']}
Reference: {item['url']}
Topic: {item['topic']}
"""

    documents.append(
        Document(
            page_content=content,
            metadata={
                "id": item["id"],
                "topic": item["topic"],
                "source": item["source"]
            }
        )
    )

print("Total documents:", len(documents))

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(
    documents,
    embedding_model
)

base_dir = os.path.dirname(os.path.abspath(__file__))
vector_path = os.path.join(base_dir, "vector_store")

vectorstore.save_local(vector_path)

print("FAISS index generated successfully!")

loaded_store = FAISS.load_local(
    vector_path,
    embedding_model,
    allow_dangerous_deserialization=True
)

print("Verified document count:", len(loaded_store.docstore._dict))

found = any(
    "Agriculture is the practice" in doc.page_content
    for doc in loaded_store.docstore._dict.values()
)

print("Agriculture definition present:", found)
