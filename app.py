from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

import json
import base64
from PIL import Image
from io import BytesIO
import pytesseract

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --------- Load data ---------
with open("tds_combined_data.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

chunks = []
for entry in raw_data:
    content = entry.get("content", "").strip()
    if len(content.split()) >= 5:
        chunks.append({
            "text": content,
            "source": entry.get("title", "Untitled"),
            "url": entry.get("original_url"),
            "type": entry.get("type", "unknown")
        })

if not chunks:
    raise RuntimeError("❌ No content loaded from tds_combined_data.json")

# --------- Embed ---------
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
texts = [c["text"] for c in chunks]
embeddings = model.encode(texts)

# --------- Setup ---------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

class QuestionInput(BaseModel):
    question: str
    image: Optional[str] = None

@app.post("/api/")
async def answer_question(payload: QuestionInput):
    query = payload.question.strip()
    if payload.image:
        image_data = base64.b64decode(payload.image)
        image = Image.open(BytesIO(image_data))
        query += " " + pytesseract.image_to_string(image)

    if not query:
        return {"error": "Empty question."}

    query_vec = model.encode([query])
    scores = cosine_similarity(query_vec, embeddings).flatten()
    top_idx = scores.argmax()
    top = chunks[top_idx]

    return {
        "question": payload.question,
        "answer": top["text"],
        "source_title": top["source"],
        "source_url": top["url"],
        "source_type": top["type"],
        "similarity_score": float(scores[top_idx])
    }

@app.get("/")
def index():
    return {"msg": "TDS Virtual TA is running."}
