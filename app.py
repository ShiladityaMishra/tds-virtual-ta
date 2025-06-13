import os

# Ensure HuggingFace downloads go to a writable local folder
os.environ["HF_HOME"] = "/data"
os.environ["TRANSFORMERS_CACHE"] = "./hf_cache"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./hf_cache"


from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json, base64
from PIL import Image
from io import BytesIO
import pytesseract

app = FastAPI()

# CORS for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data and embeddings
with open("tds_combined_data.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

chunks = []
for d in raw_data:
    text = d.get("content", "").strip()
    if len(text.split()) >= 5:
        chunks.append({
            "text": text,
            "source": d.get("title", ""),
            "url": d.get("original_url", ""),
            "type": d.get("type", "")
        })

if not chunks:
    raise RuntimeError("No content in tds_combined_data.json")




# Always use the built-in Hugging Face cache path inside Docker


# 👇 Load the token from Hugging Face secret environment
hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")

# 👇 Optional: ensure a writable cache location
os.environ["HF_HOME"] = "/data"

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    cache_folder="/data",
    token=hf_token,  # 👈 replaces use_auth_token
    trust_remote_code=True
)


texts = [c["text"] for c in chunks]
embeddings = model.encode(texts)

class QuestionInput(BaseModel):
    question: str
    image: Optional[str] = None

def extract_text_from_image(img_str):
    try:
        image = Image.open(BytesIO(base64.b64decode(img_str)))
        return pytesseract.image_to_string(image)
    except Exception:
        return ""

@app.post("/api/")
async def answer_question(q: QuestionInput):
    query = q.question.strip()
    if q.image:
        query += " " + extract_text_from_image(q.image)

    query_vec = model.encode([query])
    sims = cosine_similarity(query_vec, embeddings).flatten()
    idx = sims.argmax()
    match = chunks[idx]
    return {
        "question": q.question,
        "answer": match["text"],
        "source_title": match["source"],
        "source_url": match["url"],
        "similarity_score": float(sims[idx])
    }

@app.get("/")
def root():
    return {"msg": "API is live"}
