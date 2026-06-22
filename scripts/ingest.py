import sys
from pinecone import Pinecone, ServerlessSpec
from app.config import PINECONE_API_KEY, PINECONE_INDEX_NAME
from app.chains.embeddings import embed_text

pc = Pinecone(api_key=PINECONE_API_KEY)

if PINECONE_INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(PINECONE_INDEX_NAME)

docs = [
    {"id": "doc1", "text": "LangGraph is a library for building stateful, multi-step agent workflows."},
    {"id": "doc2", "text": "Pinecone is a managed vector database used for similarity search in RAG systems."},
    {"id": "doc3", "text": "FastAPI is a Python web framework for building APIs quickly with type hints."},
]

vectors = [(d["id"], embed_text(d["text"]), {"text": d["text"]}) for d in docs]
index.upsert(vectors=vectors)
print(f"Ingested {len(vectors)} docs into {PINECONE_INDEX_NAME}")