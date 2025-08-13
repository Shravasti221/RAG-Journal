# src/models/sentiment_model.py
from transformers import pipeline
from ..config import SENTIMENT_MODEL_NAME
from ..config import DEVICE_TYPE
from ..config import EMOTION_MODEL_NAME
# We load it lazily so it doesn't block imports
_sentiment_pipe = None

def get_sentiment_pipeline():
    """
    Returns a sentiment-analysis pipeline.
    Loads the model only once and reuses it.
    Works on CPU.
    """
    global _sentiment_pipe
    if _sentiment_pipe is None:
        _sentiment_pipe = pipeline(
            "sentiment-analysis",
            model=SENTIMENT_MODEL_NAME,
            device=DEVICE_TYPE  # Force CPU
        )
    return _sentiment_pipe
# src/models/sentiment_model.py
from transformers import pipeline

_classifier = None

def get_emotion_pipeline():
    """
    Returns a text-classification pipeline for emotion detection
    using SamLowe/roberta-base-go_emotions.
    """
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            task="text-classification",
            model=EMOTION_MODEL_NAME,
            top_k=None,   # Return all emotion scores
            device=DEVICE_TYPE    # Force CPU
        )
    return _classifier

def analyze_emotion(pipe, text):
    """
    Runs the emotion classifier on the given text.
    Returns a filtered list of dicts:
      - Always includes top 2 predictions
      - Includes any others with score >= 0.15
    """
    if not text.strip():
        return []

    # Classify and sort results
    result = pipe([text])[0]
    sorted_result = sorted(result, key=lambda x: x["score"], reverse=True)

    # Take top 2
    selected = sorted_result[:2]

    # Add any others with score >= 0.15 (and not already included)
    for emo in sorted_result[2:]:
        if emo["score"] >= 0.15:
            selected.append(emo)

    return selected

def analyze_sentiment(pipe, text):
    """
    Runs the sentiment pipeline on given text and returns
    {'label': 'POSITIVE'/'NEGATIVE', 'score': float}
    """
    if not text.strip():
        return {"label": "NEUTRAL", "score": 0.0}

    result = pipe(text)[0]
    return {"label": result["label"], "score": result["score"]}
