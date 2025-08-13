import os
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# ======== CONFIG ========
DATA_FILE = "data/diary_entries.csv"
VECTOR_STORE_DIR = "data/entries"
VECTOR_STORE_PATH = os.path.join(VECTOR_STORE_DIR, "diary_index.faiss")
METADATA_PATH = os.path.join(VECTOR_STORE_DIR, "metadata.csv")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "HuggingFaceTB/SmolLM-135M"  # Change to SmolLM v3 if available
TOP_K = 5
# ========================

def build_vector_store():
    """Build FAISS vector store from CSV."""
    try:
        diary_df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"[ERROR] No file found at {DATA_FILE}")
        return

    if diary_df.empty:
        print("[INFO] CSV is empty — nothing to index.")
        return

    if "Diary Entry" not in diary_df.columns:
        print("[ERROR] 'Diary Entry' column missing in CSV.")
        return

    if "Mood" not in diary_df.columns:
        print("[WARNING] 'Mood' column not found — metadata will not include mood.")

    print(f"[INFO] Loading embedding model: {EMBED_MODEL_NAME}")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    texts = diary_df["Diary Entry"].fillna("").tolist()
    print(f"[INFO] Generating embeddings for {len(texts)} entries...")
    embeddings = embed_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    faiss.write_index(index, VECTOR_STORE_PATH)
    diary_df.to_csv(METADATA_PATH, index=False)

    print(f"[SUCCESS] Vector store saved at {VECTOR_STORE_PATH}")
    print(f"[SUCCESS] Metadata saved at {METADATA_PATH}")

def load_vector_store():
    """Load FAISS index and metadata if available."""
    if not os.path.exists(VECTOR_STORE_PATH) or not os.path.exists(METADATA_PATH):
        print("[INFO] No existing vector store found. Building a new one...")
        build_vector_store()

    print("[INFO] Loading vector store and metadata...")
    index = faiss.read_index(VECTOR_STORE_PATH)
    diary_df = pd.read_csv(METADATA_PATH)
    return index, diary_df

def retrieve_entries(query, index, diary_df):
    """Retrieve top matching entries from FAISS store."""
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    query_emb = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices = index.search(query_emb, TOP_K)

    retrieved = []
    for idx in indices[0]:
        if 0 <= idx < len(diary_df):
            entry = diary_df.iloc[idx]["Diary Entry"][:200]  # first 200 chars
            mood = diary_df.iloc[idx]["Mood"] if "Mood" in diary_df.columns else "N/A"
            retrieved.append({"entry": entry, "mood": mood})

    return retrieved

def answer_with_llm(query, context):
    """Answer query using LLM with retrieved context."""
    print("[INFO] Loading SmolLM model...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    prompt = (
        f"Context from diary entries:\n{context}\n\n"
        f"User query: {query}\n"
        f"Answer based only on the diary entries above."
    )

    output = generator(prompt, max_new_tokens=300, do_sample=False)
    return output[0]["generated_text"]

if __name__ == "__main__":
    index, diary_df = load_vector_store()

    query = input("\nEnter your search query: ")
    retrieved_entries = retrieve_entries(query, index, diary_df)

    print("\n=== Retrieved Diary Entries (First 200 chars + Mood) ===")
    for i, item in enumerate(retrieved_entries, 1):
        print(f"{i}. Mood: {item['mood']} | {item['entry']}\n{'-'*40}")

    combined_context = "\n".join([f"[Mood: {item['mood']}] {item['entry']}" for item in retrieved_entries])
    print("__________________________________________________________________________________________________________")
    answer = answer_with_llm(query, combined_context)

    print("\n=== LLM Answer ===")
    print(answer)
