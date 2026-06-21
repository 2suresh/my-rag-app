from fastapi import FastAPI
from pydantic import BaseModel
from app.chains.graph import run_rag_graph  # your LangGraph entrypoint

app = FastAPI()


class Query(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
def query(req: Query):
    result = run_rag_graph(req.question)
    return {"answer": result}
