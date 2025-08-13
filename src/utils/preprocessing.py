# utils/preprocessing.py
import re
from typing_extensions import List, TypedDict
from langchain_core.documents import Document

def clean_text(text):
    text = re.sub(r"\s+", " ", text).strip()
    return text

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str