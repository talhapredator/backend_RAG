from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import os, json, faiss, re, uuid
from typing import List
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai
import fitz
import markdown
import numpy as np
from datetime import datetime

load_dotenv()
app = FastAPI()

# ======== Enhanced CORS ======== #
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS","https://chatbot-six-blond.vercel.app").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    expose_headers=["set-cookie"], 
    allow_headers=["*"],

    max_age=3600
)

# ======== Configuration ======== #
BASE_DIR = Path("data")
MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# ======== Core Functions ======== #
def get_session_path(session_id: str) -> Path:
    """Get or create session directory"""
    session_dir = BASE_DIR / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir

# ... [Keep all your existing functions like extract_text, generate_embeddings etc] ...
# ---------- Step 1: File Formatting ----------

def extract_text(file: UploadFile):
    suffix = file.filename.split(".")[-1].lower()
    content = ""

    if suffix == "md":
        content = markdown.markdown(file.file.read().decode("utf-8"))
    elif suffix == "txt":
        content = file.file.read().decode("utf-8")
    elif suffix == "pdf":
        doc = fitz.open(stream=file.file.read(), filetype="pdf")
        content = "\n".join([page.get_text() for page in doc])
    else:
        raise ValueError("Unsupported file format")

    return {
        "title": file.filename,
        "content": content,
        "path": file.filename
    }

# ---------- Step 2: Embedding ----------

# def generate_embeddings(articles):
#     texts = [a["content"] for a in articles]
#     embeddings = MODEL.encode(texts, convert_to_numpy=True)

#     for i, article in enumerate(articles):
#         article["embedding"] = embeddings[i].tolist()

#     return articles, embeddings

def generate_embeddings(articles):
    if not articles:
        raise ValueError("No articles provided")
        
    texts = []
    for a in articles:
        if not a.get("content"):
            continue
        texts.append(a["content"])
    
    if not texts:
        raise ValueError("No valid content found in articles")
        
    embeddings = MODEL.encode(texts, convert_to_numpy=True)
    
    # Only add embeddings for articles with content
    embedding_idx = 0
    for article in articles:
        if article.get("content"):
            article["embedding"] = embeddings[embedding_idx].tolist()
            embedding_idx += 1
            
    return articles, embeddings

# ---------- Step 3: FAISS Index ----------

# def normalize_embeddings(emb):
#     norms = np.linalg.norm(emb, axis=1, keepdims=True)
#     norms[norms == 0] = 1e-10  # Prevent division by zero
#     return emb / norms

def normalize_embeddings(emb):
    if emb is None:
        raise ValueError("Embeddings cannot be None")
    if not isinstance(emb, np.ndarray):
        emb = np.array(emb)
    if emb.size == 0:
        raise ValueError("Empty embeddings array")
        
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10  # Prevent division by zero
    return emb / norms

def build_faiss_index(embeddings):
    embeddings = embeddings.astype("float32")  # Ensure correct type
    embeddings = normalize_embeddings(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

# ---------- Step 4: Query + LLM ----------

def search_faiss(query, index, articles, top_k=3):
    q_vec = MODEL.encode([query])[0].astype("float32")
    q_vec /= np.linalg.norm(q_vec)
    D, I = index.search(np.array([q_vec]), top_k)
    return [articles[i] for i in I[0]], D[0]

def clean_html(text):
    return re.sub(r"<.*?>", "", text)

def generate_gemini_response(query, content):
    key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=key)
    context = "\n".join([a["content"] for a in content])
    clean = clean_html(context)

    prompt = f"""Answer the question based ONLY on the context:

Context:
{clean}

Question: {query}
Answer:"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    return response.text


# ======== Enhanced Endpoints ======== #
@app.middleware("http")
async def session_middleware(request: Request, call_next):
    """Auto-create session if none exists"""
    session_id = request.cookies.get("session_id") or str(uuid.uuid4())
    response = await call_next(request)
    response.set_cookie(key="session_id", value=session_id, httponly=True, secure=True,
        samesite='none',)
    return response

@app.post("/upload")
async def upload_files(request: Request, files: List[UploadFile] = File(...)):
    try:
        session_id = request.cookies.get("session_id") or str(uuid.uuid4())
        session_path = get_session_path(session_id)
        
        # Load existing articles if they exist
        articles_file = session_path / "articles.json"
        existing_articles = []
        if articles_file.exists():
            with open(articles_file, "r", encoding="utf-8") as f:
                existing_articles = json.load(f)
        
        # Process new files
        new_articles = []
        for file in files:
            try:
                article = extract_text(file)
                if article["content"]:
                    new_articles.append(article)
            except Exception as e:
                print(f"Skipping file {file.filename}: {str(e)}")
                continue
        
        if not new_articles and not existing_articles:
            raise HTTPException(status_code=400, detail="No valid files processed")
        
        # Combine old and new articles
        all_articles = existing_articles + new_articles
        
        # Generate embeddings for ALL articles (both old and new)
        all_articles, emb = generate_embeddings(all_articles)
        
        # Save updated files
        with open(str(articles_file), "w", encoding="utf-8") as f:
            json.dump(all_articles, f, indent=4, ensure_ascii=False)
        
        # Create and save new combined index
        index = build_faiss_index(np.array(emb))
        faiss.write_index(index, str(session_path / "faiss_index.index"))
        
        response = JSONResponse(
            content={
                "message": f"Added {len(new_articles)} files (Total: {len(all_articles)})",
                "session_id": session_id
            }
        )
        response.set_cookie(key="session_id", value=session_id, httponly=True, secure=True,
            samesite='none')
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/upload")
# async def upload_files(request: Request, files: List[UploadFile] = File(...)):
#     try:
#         session_id = request.cookies.get("session_id")
#         session_path = get_session_path(session_id)
        
#         articles = [extract_text(file) for file in files]
#         articles, emb = generate_embeddings(articles)
        
#         # Convert embeddings to numpy array if not already
#         if not isinstance(emb, np.ndarray):
#             emb = np.array(emb)
        
#         # Save files with explicit string paths
#         articles_file = session_path / "articles.json"
#         index_file = session_path / "faiss_index.index"
        
#         with open(str(articles_file), "w", encoding="utf-8") as f:  # Explicit str conversion
#             json.dump(articles, f, indent=4, ensure_ascii=False)
        
#         index = build_faiss_index(emb)
#         faiss.write_index(index, str(index_file))  # Explicit str conversion
        
#         return {"message": "Files processed", "session_id": session_id}
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_documents(
    request: Request,
    query: str = Form(...)
):
    try:
        session_id = request.cookies.get("session_id") or request.headers.get("x-session-id")
        if not session_id:
            raise HTTPException(status_code=401, detail="Session ID missing")
            
        session_path = get_session_path(session_id)
        
        # Validate files exist
        articles_file = session_path / "articles.json"
        index_file = session_path / "faiss_index.index"
        
        if not articles_file.exists() or not index_file.exists():
            raise HTTPException(
                status_code=400, 
                detail="Please upload documents before querying"
            )
        
        # Load articles with validation
        try:
            with open(articles_file, "r", encoding="utf-8") as f:
                articles = json.load(f)
                if not articles or not isinstance(articles, list):
                    raise ValueError("Invalid articles data")
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load articles: {str(e)}"
            )
        
        # Load FAISS index with validation
        try:
            if not os.path.getsize(index_file) > 0:
                raise ValueError("Index file is empty")
                
            index = faiss.read_index(str(index_file))
            if index.ntotal == 0:
                raise ValueError("Empty FAISS index")
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Invalid search index: {str(e)}"
            )
        
        # Validate embeddings in articles
        for article in articles:
            if "embedding" not in article or not article["embedding"]:
                raise HTTPException(
                    status_code=500,
                    detail="Some documents are missing embeddings"
                )
        
        # Perform search with additional validation
        try:
            similar, scores = search_faiss(query, index, articles)
            if not similar:
                return JSONResponse(
                    content={"response": "No relevant content found"},
                    status_code=404
                )
                
            return {"response": generate_gemini_response(query, similar)}
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Search failed: {str(e)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )