from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI()

# Load scraped discourse posts
with open("discourse_data.json", "r") as f:
    discourse_posts = json.load(f)

# Prepare text corpus
corpus_texts = [
    post['title'] + " " + " ".join(p['cooked'] for p in post['posts'])
    for post in discourse_posts
]
vectorizer = TfidfVectorizer(stop_words="english")
corpus_embeddings = vectorizer.fit_transform(corpus_texts)

class Query(BaseModel):
    question: str
    image: Optional[str] = None

@app.post("/api/")
def answer_question(query: Query):
    question_vec = vectorizer.transform([query.question])
    similarities = cosine_similarity(question_vec, corpus_embeddings).flatten()
    top_indices = similarities.argsort()[::-1][:3]

    answer = "Here's what I found:\n\n"
    links = []
    for idx in top_indices:
        post = discourse_posts[idx]
        url = f"https://discourse.onlinedegree.iitm.ac.in/t/{post['id']}"
        answer += f"- {post['title']} ([link]({url}))\n"
        links.append({"url": url, "text": post['title']})

    return {
        "answer": answer.strip(),
        "links": links
    }
