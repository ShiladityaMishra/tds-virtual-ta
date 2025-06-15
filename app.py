import os
import json
import re
import traceback
import numpy as np
import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GPT_MODEL = "gpt-4o"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading embedding model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model.encode(["test"])
print("Model loaded.")

print("Loading documents...")
try:
    with open("tds_combined_data.json", "r", encoding="utf-8") as f:
        documents = json.load(f)
except Exception as e:
    print("❌ Failed to load tds_combined_data.json")
    traceback.print_exc()
    documents = []

corpus = [doc.get("content", "") for doc in documents]
print("Encoding corpus...")
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
print("✅ Corpus encoding complete.")

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
                    {"type": "text", "text": "Please extract all text from this image."},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            }
        ],
        "max_tokens": 200
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        return result['choices'][0]['message']['content']
    else:
        print(f"OCR request failed: {response.status_code}: {response.text}")
        return ""

def generate_llm_answer(question, context):
    system_prompt = f"""
    You are a helpful assistant for the IIT Madras course 'Tools in Data Science'.
    Based on the student's question and relevant context, generate a helpful and accurate answer.

    Context:
    {context}
    """
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GPT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        "max_tokens": 400
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return "Sorry, I couldn't find a confident answer."

@app.post("/api/")
async def answer_question(request: Request):
    try:
        raw_body = await request.body()
        body_text = raw_body.decode("utf-8").strip()
        if body_text.endswith(",}"):
            body_text = body_text.replace(",}", "}")
        try:
            data = json.loads(body_text)
        except Exception as parse_err:
            return {"answer": "🚨 Invalid JSON", "links": []}

        question = data.get("question", "").strip()
        image = data.get("image")
        if not question:
            return {"answer": "Missing 'question' field.", "links": []}

        if image:
            print("Image received. Running OCR...")
            extracted_text = get_ocr(image)
            question += "\n" + extracted_text

        print(f"🔍 Question received: {question[:60]}...")
        query_embedding = model.encode(question, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=3)[0]

        if not hits:
            return {"answer": "No relevant content found.", "links": []}

        context_passages = []
        link_list = []
        for hit in hits:
            doc = documents[hit["corpus_id"]]
            content = doc.get("content", "")
            context_passages.append(content)
            url = doc.get("original_url", "")
            title = doc.get("title", "Link")
            if url:
                link_list.append({"url": url, "text": title})

        context_text = "\n---\n".join(context_passages)
        answer = generate_llm_answer(question, context_text)

        return {
            "answer": answer,
            "links": link_list or []
        }

    except Exception as e:
        print("❌ Error in /api/ endpoint")
        traceback.print_exc()
        return {"answer": "Internal server error.", "links": []}
