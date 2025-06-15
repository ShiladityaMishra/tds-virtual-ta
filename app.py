import os
import traceback
import json

# Set Hugging Face cache location
os.environ["TRANSFORMERS_CACHE"] = "/tmp"
os.environ["HF_HOME"] = "/tmp"

from fastapi import FastAPI, Request
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


@app.post("/api/")
async def answer_question(request: Request):
    try:
        # Try to safely parse JSON
        try:
            data = await request.json()
        except Exception as parse_err:
            print("❌ Could not parse JSON body")
            return {
                "question": None,
                "answer": "Invalid JSON. Probably due to template error (trailing comma, bad format).",
                "links": [],
                "debug_error": str(parse_err)
            }

        question = data.get("question", "").strip()
        image = data.get("image")  # optional, may be None

        if not question:
            return {
                "question": None,
                "answer": "No question provided in the request.",
                "links": [],
                "debug_error": "Missing 'question' field."
            }

        print(f"Received question: {question}")

        # Encode query
        query_embedding = model.encode(question, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=3)[0]
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
            "question": question,
            "answer": answer,
            "links": links
        }

    except Exception as e:
        print("❌ General error while processing question")
        traceback.print_exc()
        return {
            "question": None,
            "answer": "Server error occurred.",
            "links": [],
            "debug_error": str(e)
        }
