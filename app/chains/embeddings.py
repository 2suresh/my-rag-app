# Pinecone needs vectors — use any embedding model you like.
# This uses sentence-transformers, which runs locally on CPU with no API cost.
# The model is loaded lazily so importing this module is cheap and key-free.
from functools import lru_cache

from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    # 384-dim embeddings, free, runs on CPU. Must match the Pinecone index dim.
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed_text(text: str) -> list[float]:
    return _get_model().encode(text).tolist()
