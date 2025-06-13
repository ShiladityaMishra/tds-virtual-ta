from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import json
import os
import base64
from PIL import Image
from io import BytesIO

app = FastAPI()

# Load precomputed chunks
with open("tds_combined_data.json", "r", encoding="utf-8") as f:
    data_chunks = json.load(f)

if not data_chunks or not isinstance(data_chunks, list):
    raise RuntimeError("No valid content chunks loaded from tds_combined_data.json")

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Precompute embeddings
for chunk in data_chunks:
    chunk["embedding"] = model.encode(chunk["text"], convert_to_tensor=True)

class Query(BaseModel):
    question: str
    image: str | None = None  # base64 encoded image

@app.post("/api/")
async def query_api(payload: Query):
    question = payload.question
    image = payload.image

    if image:
        try:
            decoded_image = base64.b64decode(image)
            img = Image.open(BytesIO(decoded_image))
            question = pytesseract.image_to_string(img) + "\n" + question
        except Exception as e:
            return {"error": f"Failed to decode image: {e}"}

    # Encode the question
    question_embedding = model.encode(question, convert_to_tensor=True)

    # Compute similarity with all chunks
    results = []
    for chunk in data_chunks:
        score = util.pytorch_cos_sim(question_embedding, chunk["embedding"])[0][0].item()
        results.append({
            "text": chunk["text"],
            "similarity_score": score,
            "source_title": chunk.get("source_title", ""),
            "source_url": chunk.get("source_url", ""),
            "source_type": chunk.get("source_type", "")
        })

    # Filter out low-similarity results
    filtered_results = [r for r in results if r["similarity_score"] > 0.6]

    if not filtered_results:
        return {
            "question": question,
            "answer": "Sorry, I couldn't find a relevant answer. Please try rephrasing your question.",
            "links": []
        }

    # Get best match
    best = max(filtered_results, key=lambda x: x["similarity_score"])

    return {
        "question": question,
        "answer": best["text"],
        "source_title": best["source_title"],
        "source_url": best["source_url"],
        "source_type": best["source_type"],
        "similarity_score": best["similarity_score"]
    }

@app.get("/")
def root():
    return {"message": "TDS Virtual TA API is running!"}
