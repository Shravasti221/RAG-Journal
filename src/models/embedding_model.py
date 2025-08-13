# journal_rag/models/embedding_model.py
from sentence_transformers import SentenceTransformer

# Default model: small, fast, and decent quality on CPU
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_embedding_model = None

def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    """
    Loads and caches the embedding model for reuse.

    Args:
        model_name (str): Hugging Face model ID for sentence-transformers.

    Returns:
        SentenceTransformer: The loaded embedding model.
    """
    global _embedding_model
    if _embedding_model is None:
        print(f"Loading embedding model: {model_name}...")
        _embedding_model = SentenceTransformer(model_name)
    return _embedding_model


def embed_texts(texts):
    """
    Generates embeddings for a list of texts.

    Args:
        texts (List[str]): Input sentences/documents.

    Returns:
        List[List[float]]: Embedding vectors.
    """
    model = get_embedding_model()
    return model.encode(texts, convert_to_numpy=True)
