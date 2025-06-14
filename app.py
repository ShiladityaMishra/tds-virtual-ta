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
@app.post("/api/")
def answer_question(payload: QuestionRequest):
    try:
        print(f"Received question: {payload.question}")
        query = payload.question.strip()

        # Encode query
        query_embedding = model.encode(query, convert_to_tensor=True)
        print(f"Query embedding shape: {query_embedding.shape}")
        print(f"Corpus embedding shape: {corpus_embeddings.shape}")

        # Semantic search
        hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=3)[0]
        print(f"Top hits: {hits}")

        best = hits[0]
        best_doc = documents[best["corpus_id"]]

        answer = best_doc.get("content", "")
        url = best_doc.get("url", "")
        title = best_doc.get("title", "Link")

        links = []
        if url:
            links.append({
                "url": url,
                "text": title or "Link"
            })

        return {
            "question": query,
            "answer": answer,
            "links": links
        }

    except Exception as e:
        print("❌ Error occurred while answering question:")
        traceback.print_exc()
        return {
            "question": payload.question,
            "answer": "An error occurred while processing your question.",
            "links": [],
            "debug_error": str(e)  # helps promptfoo show the error
        }
