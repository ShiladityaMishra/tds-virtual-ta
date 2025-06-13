from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import torch
import json
import os

app = FastAPI()

# Load chunks from JSON
with open("tds_combined_data.json", "r", encoding="utf-8") as f:
    data_chunks = json.load(f)

if not data_chunks or not isinstance(data_chunks, list):
    raise RuntimeError("No valid content chunks loaded from tds_combined_data.json")

# ✅ Load model in Hugging Face Docker space with writable cache
model = SentenceTransformer("BAAI/bge-small-en-v1.5", cache_folder="/data", trust_remote_code=True)

# Precompute embeddings
for chunk in data_chunks:
    chunk["embedding"] = model.encode(chunk["text"], convert_to_tensor=True)

class Query(BaseModel):
    question: str
    image: str | None = None

@app.post("/api/")
async def query_api(payload: Query):
    question = payload.question

    # Encode the query
    question_embedding = model.encode(question, convert_to_tensor=True)

    # Compute similarities
    results = []
    for chunk in data_chunks:
        score = util.pytorch_cos_sim(question_embedding, chunk["embedding"])[0][0].item()
        results.append({
            "text": chunk["text"],
            "similarity_score": score,
            "source_title": chunk.get("source_title", ""),
            "source_url": chunk.get("source_url", "")
        })

    # Filter by similarity threshold
    filtered = [r for r in results if r["similarity_score"] > 0.6]
    if not filtered:
        return {
            "question": question,
            "answer": "Sorry, I couldn't find a relevant answer. Please rephrase or provide more details.",
            "links": []
        }

    best = max(filtered, key=lambda x: x["similarity_score"])

    return {
        "question": question,
        "answer": best["text"],
        "source_title": best["source_title"],
        "source_url": best["source_url"],
        "similarity_score": best["similarity_score"]
    }

@app.get("/")
def root():
    return {"message": "TDS Virtual TA API is running"}
