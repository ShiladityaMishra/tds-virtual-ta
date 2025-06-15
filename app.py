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
        raw_body = await request.body()
        body_text = raw_body.decode("utf-8").strip()

        # 🔧 Hack fix: remove trailing comma if it exists
        if body_text.endswith(",}"):
            body_text = body_text.replace(",}", "}")

        try:
            data = json.loads(body_text)
        except json.JSONDecodeError as e:
            return {
                "question": None,
                "answer": "🚨 Invalid JSON received (possibly due to Promptfoo templating error).",
                "links": [],
                "debug_error": str(e)
            }

        question = data.get("question", "").strip()
        if not question:
            return {
                "question": None,
                "answer": "No question received.",
                "links": [],
                "debug_error": "Missing 'question' field"
            }

        print(f"Received question: {question}")
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

    except Exception as e:
        print("❌ Unhandled error:")
        traceback.print_exc()
        return {
            "question": None,
            "answer": "Server crashed. See debug log.",
            "links": [],
            "debug_error": str(e)
        }
