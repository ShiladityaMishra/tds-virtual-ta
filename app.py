import os

# ✅ Redirect Hugging Face model cache to /tmp
os.environ["TRANSFORMERS_CACHE"] = "/tmp"
os.environ["HF_HOME"] = "/tmp"

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import json
from sentence_transformers import SentenceTransformer, util



app = FastAPI()

# Load model (uses /tmp for cache)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load your scraped + combined data
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
    return {
        "question": query,
        "answer": best_doc["content"],
        "source_title": best_doc.get("title", ""),
        "source_url": best_doc.get("url", ""),
        "similarity_score": float(best["score"])
    }
