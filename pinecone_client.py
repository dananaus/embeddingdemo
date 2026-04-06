"""
Pinecone index management and upsert/query helpers.
"""

from pinecone import Pinecone, ServerlessSpec

from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_ENVIRONMENT,
    EMBEDDING_DIMENSIONS,
)

_pc = Pinecone(api_key=PINECONE_API_KEY)


def get_or_create_index():
    """Return the Pinecone index, creating it if it doesn't exist."""
    existing = [i.name for i in _pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        print(f"Creating Pinecone index '{PINECONE_INDEX_NAME}' ({EMBEDDING_DIMENSIONS}d, cosine)...")
        _pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSIONS,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT),
        )
        print("Index created.")
    return _pc.Index(PINECONE_INDEX_NAME)


def upsert_vector(index, vector_id: str, embedding: list[float], metadata: dict):
    """Upsert a single vector with metadata."""
    index.upsert(vectors=[{"id": vector_id, "values": embedding, "metadata": metadata}])


def upsert_batch(index, vectors: list[dict]):
    """
    Upsert a batch of vectors.
    Each dict: {"id": str, "values": list[float], "metadata": dict}
    """
    index.upsert(vectors=vectors)


def query_index(index, query_embedding: list[float], top_k: int = 5, filter: dict = None):
    """Query the index and return top_k matches with metadata."""
    kwargs = {"vector": query_embedding, "top_k": top_k, "include_metadata": True}
    if filter:
        kwargs["filter"] = filter
    return index.query(**kwargs)
