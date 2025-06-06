from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import base64
from sentence_transformers import SentenceTransformer, util
import json

app = FastAPI()

# Load scraped data
with open("discourse_data.json", "r") as f:
    discourse_posts = json.load(f)

# Load embedding model
model = SentenceTransformer('paraphrase-albert-small-v2')  # smaller model to save space
corpus = [post['title'] + " " + " ".join(p['cooked'] for p in post['posts']) for post in discourse_posts]
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

class Query(BaseModel):
    question: str
    image: Optional[str] = None

@app.post("/api/")
def answer_question(query: Query):
    question_embedding = model.encode(query.question, convert_to_tensor=True)
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=3)[0]

    answer = "Here's what I found:\n\n"
    links = []
    for hit in hits:
        idx = hit['corpus_id']
        post = discourse_posts[idx]
        url = f"https://discourse.onlinedegree.iitm.ac.in/t/{post['id']}"
        answer += f"- {post['title']} ([link]({url}))\n"
        links.append({"url": url, "text": post['title']})

    return {
        "answer": answer.strip(),
        "links": links
    }
