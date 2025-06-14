import os

# Redirect cache folders to writable location
os.environ["TRANSFORMERS_CACHE"] = "/tmp"
os.environ["HF_HOME"] = "/tmp"

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import json
from sentence_transformers import SentenceTransformer, util
model.encode(["test"])  # warm-up

app = FastAPI()

# Load model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load documents
with open("tds_combined_data.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

corpus = [doc["content"] for doc in documents]
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

class QuestionRequest(BaseModel):
    question: str
    image: Optional[str] = None

@app.post("/api/")
def answer_question(payload: QuestionRequest):
    query = payload.question.strip()
    query_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=3)[0]
    best = hits[0]
    best_doc = documents[best["corpus_id"]]

    # Create answer
    answer = best_doc.get("content", "")

    # Add links (empty if not found)
    url = best_doc.get("url", "")
    title = best_doc.get("title", "Link")

    links = []
    if url:
        links.append({
            "url": url,
            "text": title or "Link"
        })

    return {
        "answer": answer,
        "links": links
    }
