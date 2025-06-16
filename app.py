import os

# Set HF cache directory to writable /tmp path
os.environ["TRANSFORMERS_CACHE"] = "/tmp"
os.environ["HF_HOME"] = "/tmp"

import json
import re
import traceback
import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from PIL import Image
import pytesseract
import base64
from io import BytesIO

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str = None
    image: str = None  # optional base64 string

# ✅ Load data
print("Loading documents...")
try:
    with open("tds_combined_data.json", "r", encoding="utf-8") as f:
        documents = json.load(f)
except Exception as e:
    print("❌ Failed to load tds_combined_data.json")
    traceback.print_exc()
    documents = []

corpus = [doc.get("content", "") for doc in documents]

# ✅ Embed corpus
print("Encoding corpus...")
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embedding_model.encode(["test"])
corpus_embeddings = embedding_model.encode(corpus, convert_to_tensor=True)
print("✅ Corpus ready")

# ✅ Load LLM
print("Loading LLM pipeline...")
llm = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", device=0)
print("✅ LLM ready")

def get_ocr(image_data):
    try:
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        return pytesseract.image_to_string(image)
    except Exception as e:
        print(f"❌ OCR error: {e}")
        return ""

def ask_llm(question, context):
    try:
        prompt = f"Question: {question}\nContext: {context}\nAnswer:"
        response = llm(prompt, max_new_tokens=200, do_sample=False)[0]["generated_text"]
        return response.split("Answer:")[-1].strip()
    except Exception as e:
        print(f"❌ LLM error: {e}")
        return "Sorry, I could not generate an answer."

@app.post("/api/")
async def answer_query(query: QueryRequest):
    try:
        if not query or not isinstance(query.question, str):
            return {"answer": "Invalid request: 'question' is required.", "links": []}

        question = query.question.strip()
        if not question:
            return {"answer": "Missing question.", "links": []}

        if query.image:
            print("📷 Image received, extracting text...")
            ocr_text = get_ocr(query.image)
            question += "\n" + ocr_text

        embedding = embedding_model.encode(question, convert_to_tensor=True)

        # Find top 3 matches
        hits = util.semantic_search(embedding, corpus_embeddings, top_k=3)[0]
        context = "\n---\n".join([documents[h["corpus_id"]].get("content", "") for h in hits])
        links = []
        for h in hits:
            doc = documents[h["corpus_id"]]
            url = doc.get("original_url", "")
            title = doc.get("title", "Link")
            if url:
                links.append({"url": url, "text": title})

        answer = ask_llm(question, context)

        return {
            "answer": answer or "No answer generated.",
            "links": links or []
        }

    except Exception as e:
        print("❌ Exception:", e)
        traceback.print_exc()
        return {"answer": "An error occurred while processing your request.", "links": []}
