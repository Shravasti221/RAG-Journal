# rag/vector_store.py
import os
import json
from langchain import hub
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from typing import Any
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from ..models.embedding_model import get_embedding_model
from ..config import VECTOR_STORE_PATH
from ..utils.preprocessing import State


class JournalVectorStore:
    def __init__(self, llm, embedding_model, vector_store_path=VECTOR_STORE_PATH):
        self.vector_store_path = vector_store_path
        self.embedding_model = embedding_model
        self.vector_store = self._load_vector_store()
        self.prompt = hub.pull("rlm/rag-prompt")
        self.llm = llm

    def _load_vector_store(self):
        """Load vector store from disk or initialize a new one."""
         # Step 1: Resolve path one folder above current file
        base_dir = os.path.dirname(os.path.abspath(__file__))  # current file dir
        parent_dir = os.path.dirname(base_dir)  # one level up
        vector_store_file = os.path.join(parent_dir, VECTOR_STORE_PATH)
        print("Attempting to access vector store from : ", vector_store_file)

        # Step 2: Ensure directory exists
        os.makedirs(os.path.dirname(vector_store_file), exist_ok=True)

        # Step 3: If file does not exist, create empty JSON file
        if not os.path.exists(vector_store_file):
            with open(vector_store_file, "w", encoding="utf-8") as f:
                json.dump({}, f)  # empty JSON object
            print(f"Created empty vector store at {vector_store_file}")
            return None  # or return empty store object if preferred

        # Step 4: Load vector store from file
        try:
            vector_store = InMemoryVectorStore.load(
                path=vector_store_file,
                embedding=self.embedding_model
            )
            print(f"Loaded vector store from {vector_store_file}")
            return vector_store
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return None

    def add_entry(self, entry_text, metadata=None):
        """Add a journal entry to the vector store."""
        if metadata is None:
            metadata = {}
        doc = Document(page_content=entry_text, metadata=metadata)
        self.vector_store.add_documents([doc])
        self._save_vector_store()

    def search_entries(self, query, k=5):
        """Retrieve top-k similar journal entries for the given query."""
        print("Type of vector store: ", self.vector_store, type(self.vector_store))
        print("Querying vector store for:", query)
        return self.vector_store.similarity_search(query, k=k)
    
    def retrieve(self, state: State) -> List[Document]:
        """Retrieve documents from the vector store based on the state."""
        retrieved_docs = self.vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def generate_response(self, state: State) -> dict:
        """Generate a response based on the state."""
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.prompt.invoke({"question": state["question"], "context": docs_content})
        response = self.llm.invoke(messages)
        return {"answer": response}

    def chat(self, question: str) -> str:
        """Handle a chat interaction based on the state."""
        graph_builder = StateGraph(State).add_sequence([self.retrieve, self.generate_response])
        graph_builder.add_edge(START, "retrieve")
        graph = graph_builder.compile()
        return graph.invoke({"question": question})["answer"]