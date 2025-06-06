import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import openai
from scipy.spatial.distance import cosine

# Load your OpenAI API key from Replit secrets (will set up next)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load discourse data
with open("discourse_data.json", "r") as f:
    data = json.load(f)

discourse_posts = data['discourse']

# Prepare corpus texts by joining title + all posts cooked text
corpus = [
    post['title'] + " " + " ".join(p['cooked'] for p in post['posts'])
    for post in discourse_posts
]

# Function to get OpenAI embedding for a given text
def get_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

# Generate embeddings for all corpus texts
print("Generating embeddings for corpus... This may take a while.")
corpus_embeddings = [get_embedding(text) for text in corpus]
print("Embeddings ready!")

app = FastAPI()

class Query(BaseModel):
    question: str
    image: Optional[str] = None

@app.post("/api/")
def answer_question(query: Query):
    question_embedding = get_embedding(query.question)

    # Calculate similarity with each corpus embedding
    similarities = [(1 - cosine(question_embedding, emb), idx) for idx, emb in enumerate(corpus_embeddings)]
    similarities.sort(reverse=True)
    top_k = similarities[:3]  # top 3 matches

    answer = "Here's what I found:\n\n"
    links = []
    for score, idx in top_k:
        post = discourse_posts[idx]
        url = post.get('url', f"https://discourse.onlinedegree.iitm.ac.in/t/{post['id']}")
        answer += f"- {post['title']} ([link]({url}))\n"
        links.append({"url": url, "text": post['title']})

    return {
        "answer": answer.strip(),
        "links": links
    }
