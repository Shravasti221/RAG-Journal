# src/app.py
import streamlit as st
from langchain import hub
from src.rag.vector_store import JournalVectorStore
from src.models.load_model import load_local_llm
from src.models.sentiment_model import get_emotion_pipeline, analyze_emotion
from src.models.topic_model import get_topic_model, extract_topics, update_topic_model
from src.models.embedding_model import get_embedding_model
from src.utils.preprocessing import clean_text

# Page title
st.set_page_config(page_title="Reflective Journal RAG", layout="wide")
st.title("📓 Reflective Journal RAG")

# Load models
llm_pipe = load_local_llm()
emotion_pipe = get_emotion_pipeline()
topic_model = get_topic_model()
embedding_model = get_embedding_model()
# Load vector store
store = JournalVectorStore(llm=llm_pipe, embedding_model = embedding_model)

# Sidebar Navigation
st.sidebar.title("Navigation")
mode = st.sidebar.radio(
    "Choose an option:",
    ["Add Entry", "Retrieve Entries", "Chat with Me"]
)

# --- 1) Add Entry ---
if mode == "Add Entry":
    st.header("Add a New Journal Entry")
    mood = st.text_input("Mood")
    entry = st.text_area("Write your journal entry here:")

    if st.button("Save Entry"):
        if not entry.strip():
            st.warning("Please write something before saving.")
        else:
            cleaned = clean_text(entry)

            # Run emotion classification (used as mood if mood not given)
            if not mood:
                emotions_list = analyze_emotion(emotion_pipe, cleaned)
                if emotions_list:
                    mood = emotions_list[0]['label']

            # Topic extraction
            topics = extract_topics(topic_model, cleaned)
            update_topic_model(topic_model, [cleaned])

            metadata = {
                "mood": mood,
                "topics": topics
            }

            store.add_entry(cleaned, metadata)
            st.success(f"Entry saved with mood '{mood}' and topics {topics}.")

# --- 2) Retrieve Entries ---
elif mode == "Retrieve Entries":
    st.header("Retrieve Journal Entries")
    query = st.text_input("Enter a search query:")

    if st.button("Retrieve"):
        if not query.strip():
            st.warning("Please enter a query.")
        else:
            retrieved_docs = store.search_entries(query, k=5)
            if not retrieved_docs:
                st.info("No matching entries found.")
            else:
                st.subheader("Retrieved Past Entries:")
                for idx, doc in enumerate(retrieved_docs, start=1):
                    st.write(f"**Entry {idx}:** {doc.page_content}")
                    st.caption(f"Mood: {doc.metadata.get('mood', 'N/A')}")
                    if 'topics' in doc.metadata:
                        st.caption(f"Topics: {', '.join(doc.metadata['topics'])}")

# --- 3) Chat with Me ---
elif mode == "Chat with Me":
    st.header("Chat with Your Journal")
    question = st.text_input("Ask a question about your past entries:")

    if st.button("Ask"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            answer = store.chat(question)
            st.subheader("Answer:")
            st.write(answer)
