import os

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from knowledge_base import KNOWLEDGE_BASE

print("Preparing knowledge documents...")

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

print("Creating FAISS vector store...")

vectorstore = FAISS.from_documents(
    documents,
    embedding_model
)

# Save in the same directory as this Python file
base_dir = os.path.dirname(os.path.abspath(__file__))
vector_path = os.path.join(base_dir, "vector_store")

vectorstore.save_local(vector_path)

print("FAISS index saved successfully!")

# Test agriculture retrieval
print("\nTesting agriculture retrieval...")

results = vectorstore.similarity_search(
    "What is agriculture?",
    k=5
)

for doc in results:
    print("\nID:", doc.metadata.get("id"))
    print("Topic:", doc.metadata.get("topic"))
    print("Content:", doc.page_content)

print("\nTotal indexed documents:", len(vectorstore.docstore._dict))
