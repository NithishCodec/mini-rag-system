from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from google import genai
import os
from similarity import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware

# -------------------
# App
# -------------------
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------
# Gemini client
# -------------------
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
model = "models/text-embedding-004"
GEN_MODEL = "models/gemini-2.5-flash"

# -------------------
# In-memory storage
# -------------------
sentences = []
embeddings = []

# -------------------
# Schemas
# -------------------
class SentenceList(BaseModel):
    sentences: list[str]  # kept (backward compatible)

class TextInput(BaseModel):
    text: str  # NEW: raw paragraph input

class Query(BaseModel):
    question: str

# -------------------
# Chunking helper
# -------------------
def chunk_text(text: str):
    # basic sentence-level chunking
    chunks = text.replace("\n", " ").split(".")
    return [c.strip() for c in chunks if c.strip()]




@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")

# -------------------
# Add sentences endpoint (OLD - kept)
# -------------------
@app.post("/add")
def add_sentences(data: SentenceList):
    global sentences, embeddings

    response = client.models.embed_content(
        model=model,
        contents=data.sentences
    )

    for i, emb in enumerate(response.embeddings):
        sentences.append(data.sentences[i])
        embeddings.append(emb.values)

    return {
        "message": "Sentences added successfully",
        "total_sentences": len(sentences)
    }

# -------------------
# Add paragraph endpoint (NEW)
# -------------------
@app.post("/add_text")
def add_text(data: TextInput):
    global sentences, embeddings

    chunks = chunk_text(data.text)

    response = client.models.embed_content(
        model=model,
        contents=chunks
    )

    for i, emb in enumerate(response.embeddings):
        sentences.append(chunks[i])
        embeddings.append(emb.values)

    return {
        "message": "Text added successfully",
        "chunks_added": len(chunks),
        "total_chunks": len(sentences)
    }

# -------------------
# Search endpoint
# -------------------
@app.post("/search")
def search(query: Query):
    if not sentences:
        return {"error": "No sentences added yet"}

    # ---- Embed query ----
    query_embedding = client.models.embed_content(
        model=model,
        contents=query.question
    ).embeddings[0].values

    # ---- Similarity search ----
    scores = []
    for i, emb in enumerate(embeddings):
        score = cosine_similarity(query_embedding, emb)
        scores.append((sentences[i], score))

    scores.sort(key=lambda x: x[1], reverse=True)

    # ---- Confidence gate ----
    TOP_K = 3
    THRESHOLD = 0.35

    top_matches = [(s, sc) for s, sc in scores[:TOP_K] if sc >= THRESHOLD]

    if not top_matches:
        return {
            "query": query.question,
            "answer": "I don't know based on the provided data."
        }

    # ---- Build context ----
    context = "\n".join([f"- {s}" for s, _ in top_matches])

    # ---- Controlled prompt ----
    prompt = f"""
You are an assistant that answers questions using ONLY the information below.
Do not use any external knowledge.
If the answer is not present, say "I don't know".

Information:
{context}

Question:
{query.question}

Answer:
"""

    # ---- Generate answer ----
    response = client.models.generate_content(
        model=GEN_MODEL,
        contents=prompt
    )

    return {
        "query": query.question,
        "answer": response.text.strip(),
        "sources": [s for s, _ in top_matches]
    }
