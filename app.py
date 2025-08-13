# src/app.py
import streamlit as st
import pandas as pd
from langchain import hub
from src.rag.vector_store import JournalVectorStore
from src.models.load_model import load_local_llm, generate_response
from src.models.sentiment_model import get_emotion_pipeline, analyze_emotion
from src.models.topic_model import get_topic_model, extract_topics, update_topic_model
from src.utils.preprocessing import clean_text, State
from src.config import DATA_FILE


# Load models
st.title("📓 Reflective Journal RAG")
llm_pipe = load_local_llm()
emotion_pipe = get_emotion_pipeline()
topic_model = get_topic_model()

# Load vector store
store = JournalVectorStore(llm=llm_pipe)
# Load existing data
try:
    diary_df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    diary_df = pd.DataFrame(columns=["Date", "Mood", "Diary Entry", "Emotions", "Topics"])

# Sidebar navigation
mode = st.sidebar.selectbox("Mode", ["Add Entry", "Search Journal"])

if mode == "Add Entry":
    mood = st.text_input("Mood")
    entry = st.text_area("Write your diary entry")

    if st.button("Save Entry"):
        cleaned = clean_text(entry)

        # Run emotion classification
        emotions_list = analyze_emotion(emotion_pipe, cleaned)
        formatted_emotions = ", ".join([f"{e['label']} ({e['score']:.2f})" for e in emotions_list])

        # Topic extraction
        topics = extract_topics(topic_model, cleaned)
        update_topic_model(topic_model, [cleaned])

        metadata = {
            "mood": mood,
            "emotions": formatted_emotions,
            "topics": topics
        }

        store.add_entry(cleaned, metadata)

        diary_df = diary_df.append({
            "Date": pd.Timestamp.now().strftime("%Y-%m-%d"),
            "Mood": mood,
            "Diary Entry": cleaned,
            "Emotions": formatted_emotions,
            "Topics": ", ".join(topics)
        }, ignore_index=True)

        diary_df.to_csv(DATA_FILE, index=False)
        st.success("Entry saved with emotions and topics.")

elif mode == "Search Journal":
    query = st.text_input("Search or ask a question")

    if st.button("Search"):
        retrieved_docs = store.search_entries(query)
        st.write("**Retrieved Past Entries:**")
        for doc in retrieved_docs:
            st.write(
                f"- {doc.page_content} "
                f"(Emotions: {doc.metadata.get('emotions', 'N/A')}, "
                f"Topics: {doc.metadata.get('topics', [])})"
            )

        if st.button("Generate Insight"):
            answer = generate_response(llm_pipe, query, retrieved_docs)
            st.write("**Insight:**")
            st.write(answer)
