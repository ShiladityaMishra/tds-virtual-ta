import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from sentence_transformers import SentenceTransformer, util

# ✅ Safe cache paths for HF Spaces
os.environ["TRANSFORMERS_CACHE"] = "/tmp"
os.environ["HF_HOME"] = "/tmp"

app = FastAPI()

# ✅ Load model and corpus
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
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

    best_hit = hits[0]
    best_doc = documents[best_hit["corpus_id"]]

    answer = best_doc["content"]
    url = best_doc.get("url", "")
    title = best_doc.get("title", "Relevant Discussion")

    # ✅ Always return a list of links, even if empty
    links = [{"url": url, "text": title}] if url else []

    return {
        "question": query,
        "answer": answer,
        "links": links
    }
