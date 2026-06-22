from typing import TypedDict, List
from functools import lru_cache

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from pinecone import Pinecone

from app.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    GROQ_API_KEY,
    GROQ_MODEL,
)

# --- Clients (lazy) ---
# Created on first use, not at import time, so the app and tests can load
# without valid API keys present (keys are only required to actually answer).


@lru_cache(maxsize=1)
def get_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX_NAME)


@lru_cache(maxsize=1)
def get_llm():
    return ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.3)


# --- State ---


class RAGState(TypedDict):
    question: str
    context: List[str]
    answer: str


# --- Nodes ---


def retrieve(state: RAGState) -> RAGState:
    # Embed the query — swap in your real embedding call
    from app.chains.embeddings import embed_text
    vector = embed_text(state["question"])

    results = get_index().query(
        vector=vector, top_k=4, include_metadata=True)
    chunks = [match["metadata"].get("text", "")
              for match in results["matches"]]
    return {**state, "context": chunks}


def generate(state: RAGState) -> RAGState:
    context_block = "\n\n".join(
        state["context"]) if state["context"] else "No context found."
    prompt = (
        f"Answer the question using only the context below. "
        f"If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{context_block}\n\nQuestion: {state['question']}"
    )
    response = get_llm().invoke(prompt)
    return {**state, "answer": response.content}


# --- Graph ---


@lru_cache(maxsize=1)
def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def run_rag_graph(question: str) -> str:
    result = build_graph().invoke(
        {"question": question, "context": [], "answer": ""})
    return result["answer"]
