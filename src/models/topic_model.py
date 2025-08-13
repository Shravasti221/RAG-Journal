# src/models/topic_model.py
import os
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

TOPIC_MODEL_PATH = "data/topic_model"

_topic_model = None

def get_topic_model():
    """
    Loads (or initializes) a BERTopic model with MiniLM embeddings for topic extraction.
    Model is persisted to disk to maintain knowledge across runs.
    """
    global _topic_model

    if _topic_model is not None:
        return _topic_model

    if os.path.exists(TOPIC_MODEL_PATH):
        print(f"[INFO] Loading existing BERTopic model from {TOPIC_MODEL_PATH}")
        _topic_model = BERTopic.load(TOPIC_MODEL_PATH)
    else:
        print("[INFO] Creating new BERTopic model")
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        _topic_model = BERTopic(embedding_model=embedding_model)

    return _topic_model


def extract_topics(model, text, top_n=3):
    """
    Extracts top topics for the given text.
    If model has no prior data for this, it updates incrementally.
    Returns a list of topic words.
    """
    if not text.strip():
        return []

    topics, _ = model.transform([text])

    # If topic assignment fails (-1), we need to fit this text
    if topics[0] == -1:
        model.partial_fit([text])
        topics, _ = model.transform([text])

    topic_info = model.get_topic(topics[0])
    if topic_info is None:
        return []

    # Extract top_n words for the topic
    extracted = [word for word, _ in topic_info[:top_n]]
    return extracted


def update_topic_model(model, texts):
    """
    Updates the BERTopic model with new texts.
    Persists the updated model to disk.
    """
    if not texts:
        return

    model.partial_fit(texts)
    model.save(TOPIC_MODEL_PATH)
