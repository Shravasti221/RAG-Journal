# rag/vector_store.py
import pickle
from langchain import hub
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from ..models.embedding_model import get_embedding_model
from ..config import VECTOR_STORE_PATH
from ..utils.preprocessing import State


class JournalVectorStore:
    def __init__(self, llm, vector_store_path=VECTOR_STORE_PATH):
        self.vector_store_path = vector_store_path
        self.embeddings = get_embedding_model()
        self.vector_store = self._load_vector_store()
        self.prompt = hub.pull("rlm/rag-prompt")
        self.llm = llm

    def _load_vector_store(self):
        """Load vector store from disk or initialize a new one."""
        try:
            with open(self.vector_store_path, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return InMemoryVectorStore(self.embeddings)

    def _save_vector_store(self):
        """Save current vector store to disk."""
        with open(self.vector_store_path, "wb") as f:
            pickle.dump(self.vector_store, f)

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