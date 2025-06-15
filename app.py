import os

# Set Hugging Face cache location
os.environ["TRANSFORMERS_CACHE"] = "/tmp"
os.environ["HF_HOME"] = "/tmp"

import traceback
import json

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from sentence_transformers import SentenceTransformer, util


# Initialize FastAPI app
app = FastAPI()

# Load model
print("Loading model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model.encode(["test"])  # Warm-up
print("Model loaded.")

# Load documents
print("Loading documents...")
try:
    with open("tds_combined_data.json", "r", encoding="utf-8") as f:
        documents = json.load(f)
except Exception as e:
    print("❌ Failed to load `tds_combined_data.json`")
    traceback.print_exc()
    documents = []

print(f"✅ Loaded {len(documents)} documents")

# Encode corpus
corpus = [doc.get("content", "") for doc in documents]
print("Encoding corpus...")
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
print("✅ Corpus encoding complete.")

# Define input format
class QuestionRequest(BaseModel):
    question: str
    image: Optional[str] = None

# API endpoint
from fastapi import Request

@app.post("/api/")
async def answer_question(request: Request):
    try:
        # Manually parse raw JSON to catch errors from malformed body
        raw = await request.body()
        data = json.loads(raw)

        question = data.get("question", "").strip()
        image = data.get("image", None)

        print(f"Received question: {question}")

        # Proceed as usual
        query_embedding = model.encode(question, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=3)[0]
        best = hits[0]
        best_doc = documents[best["corpus_id"]]

        answer = best_doc.get("content", "")
        url = best_doc.get("url", "")
        title = best_doc.get("title", "Link")

        links = []
        if url:
            links.append({"url": url, "text": title})

        return {
            "question": question,
            "answer": answer,
            "links": links
        }

    except json.JSONDecodeError as jde:
        print("❌ Invalid JSON received.")
        return {
            "question": None,
            "answer": "Invalid JSON body sent to API. Please fix the formatting.",
            "links": [],
            "debug_error": str(jde)
        }
    except Exception as e:
        print("❌ General error:")
        traceback.print_exc()
        return {
            "question": None,
            "answer": "Unexpected server error occurred.",
            "links": [],
            "debug_error": str(e)
        }
