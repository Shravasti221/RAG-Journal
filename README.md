# 📝 Local Digital Journal (RAG-powered)

A **local, privacy-focused digital journal** built with **LangChain Retrieval-Augmented Generation (RAG)**.
Keep your thoughts secure on your own machine, while being able to **search, chat with your past self, and generate meaningful insights** from your entries.

---

## ✨ Features

* **🔒 Local & Private** – All data is stored and processed locally for complete privacy.
* **📥 Add Entries** – Log your thoughts, daily reflections, or notes.
* **🔍 Semantic Search** – Find past entries using **vector embeddings** instead of simple keyword search.
* **💬 Chat with Your Past Self** – Ask questions and retrieve relevant memories from your journal.
* **🧠 Context-Aware Insights** – Uses RAG to combine search results with LLM reasoning for richer responses.

---

## 🛠️ Tech Stack

* **LangChain** – Orchestration for LLM + Retrieval
* **FAISS** – Local vector database for embeddings
* **SentenceTransformers / mpnet Embeddings** – For semantic search
* **Streamlit / CLI** – Simple interface for journaling and chatting
* **LLM (Local or API)** – SmolLMv3/ Gemma, or other supported models

### 1️⃣ Install dependencies

```bash
uv sync
```

### 2️⃣ Run the app

```bash
uv run streamlit run app.py
```
