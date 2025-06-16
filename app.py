import os
import json
import re
# ✅ Avoid HF Spaces permission errors
os.environ["TRANSFORMERS_CACHE"] = "/tmp"
os.environ["HF_HOME"] = "/tmp"
import traceback
import numpy as np
import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv

# ✅ Load OpenAI API key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API")
GPT_MODEL = "gpt-4o"


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
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model.encode(["test"])
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
print("✅ Corpus ready")

def get_ocr(image_data):
    data_url = f"data:image/webp;base64,{image_data}"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GPT_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all text from this image."},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            }
        ]
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    print(f"❌ OCR failed: {response.status_code}: {response.text}")
    return ""

def ask_llm(question, context):
    system_prompt = f"""
    You are a TA for IITM's Tools in Data Science course. Below is the student's query and relevant past material. Respond helpfully and concisely.

    Context:
    {context}
    """
    payload = {
        "model": GPT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    }
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    print(f"❌ LLM call failed: {response.status_code} - {response.text}")
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

        # Embed query
        payload = {
            "model": "text-embedding-3-small",
            "input": [question]
        }
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        resp = requests.post("https://api.openai.com/v1/embeddings", headers=headers, json=payload)

        if resp.status_code != 200:
            print(f"❌ Embedding API failed: {resp.status_code} — {resp.text}")
            return {"answer": "Failed to process embedding request.", "links": []}

        embedding = np.array(resp.json()["data"][0]["embedding"])

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