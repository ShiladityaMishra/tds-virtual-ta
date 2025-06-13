from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import json
import base64
from io import BytesIO
from PIL import Image
import os

from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer, util

app = FastAPI()

# Step 1: Download model to a writable directory
model_path = snapshot_download(repo_id="sentence-transformers/all-MiniLM-L6-v2", local_dir="./model", ignore_patterns=["*.msgpack"])
model = SentenceTransformer(model_path)

# Step 2: Load your preprocessed data
with open("tds_combined_data.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

# Step 3: Prepare corpus and embeddings
corpus = [doc["content"] for doc in documents]
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

# Step 4: Define input schema
class QuestionRequest(BaseModel):
    question: str
    image: Optional[str] = None  # base64-encoded image (optional)

# Step 5: Define the API route
@app.post("/api/")
def answer_question(payload: QuestionRequest):
    query = payload.question.strip()

    # Embed and search
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
