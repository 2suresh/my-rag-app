# Pinecone needs vectors. We use fastembed: it runs the same all-MiniLM-L6-v2
# model as sentence-transformers but via lightweight ONNX runtime (no PyTorch),
# so it fits inside Render's free 512 MB tier. Output is still 384-dim, so the
# existing Pinecone index stays compatible.
# The model is loaded lazily so importing this module is cheap and key-free.
from functools import lru_cache

from fastembed import TextEmbedding


@lru_cache(maxsize=1)
def _get_model() -> TextEmbedding:
    # 384-dim embeddings, free, low-memory CPU. Must match the Pinecone index dim.
    return TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")


def embed_text(text: str) -> list[float]:
    # .embed() returns a generator of numpy arrays; take the first.
    return next(_get_model().embed([text])).tolist()
