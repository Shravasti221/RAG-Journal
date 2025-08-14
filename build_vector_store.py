import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
# ======== CONFIG ========
EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL_NAME = "HuggingFaceTB/SmolLM-135M"  # Change to SmolLM v3 if available
TOP_K = 5
# ========================

# Initialize embeddings and vector store
print(f"[INFO] Loading embedding model: {EMBED_MODEL_NAME}")
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
vector_store = InMemoryVectorStore(embeddings)
diary_df = pd.read_csv("C:\Users\Admin\source\Journal\data\diary_entries.csv")
all_splits = [Document(page_content=text) for text in diary_df['Diary Entry'].dropna()]
document_ids = vector_store.add_documents(documents=all_splits)


def add_entry(entry_text, mood):
    """Add a new diary entry to the in-memory vector store."""
    metadata = {"mood": mood}
    vector_store.add_texts([entry_text], metadatas=[metadata])
    print("[SUCCESS] Entry added to vector store.")

def retrieve_entries(query):
    """Retrieve top matching entries from the in-memory store."""
    results = vector_store.similarity_search(query, k=TOP_K)
    retrieved = []
    for doc in results:
        entry_snippet = doc.page_content[:200]  # First 200 chars
        mood = doc.metadata.get("mood", "N/A")
        retrieved.append({"entry": entry_snippet, "mood": mood})
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
    # Example usage
    while True:
        action = input("\nChoose action: [1] Add Entry, [2] Search, [3] Chat, [q] Quit: ").strip()
        if action == "1":
            mood = input("Mood: ")
            entry = input("Diary entry: ")
            add_entry(entry, mood)
        elif action == "2":
            query = input("Search query: ")
            retrieved_entries = retrieve_entries(query)
            print("\n=== Retrieved Diary Entries ===")
            for i, item in enumerate(retrieved_entries, 1):
                print(f"{i}. Mood: {item['mood']} | {item['entry']}\n{'-'*40}")
        elif action == "3":
            query = input("Ask me something: ")
            retrieved_entries = retrieve_entries(query)
            combined_context = "\n".join([f"[Mood: {item['mood']}] {item['entry']}" for item in retrieved_entries])
            answer = answer_with_llm(query, combined_context)
            print("\n=== LLM Answer ===")
            print(answer)
        elif action.lower() == "q":
            break
