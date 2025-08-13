# config.py

# Embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# Sentiment model
SENTIMENT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english" #"cardiffnlp/twitter-roberta-base-sentiment"

#Emotion model
EMOTION_MODEL_NAME = "SamLowe/roberta-base-go_emotions"

# Local LLM model
LOCAL_LLM_MODEL_ID = "HuggingFaceTB/SmolLM-135M"

# Vector store persistence file
VECTOR_STORE_PATH = "data/vector_store.pkl"

# Data file
DATA_FILE = "data/diary_entries.csv"

# Max retrieved docs
TOP_K = 5

DEVICE_TYPE = "cpu"  # Force CPU for inference
# DEVICE_TYPE = "cuda"  # Uncomment to use GPU if available
