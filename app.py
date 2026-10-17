import os
import sys

# 1. Path Setup: Current folder ko Python environment me explicitly set karna
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

# Local file import
import proces_incoming

app = FastAPI(title="Java RAG Assistant API")

# 2. Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Load Embeddings Data at Startup
try:
    df = joblib.load('embeddings.joblib')
    doc_embeddings = np.vstack(df['embedding'].values)
    print("SUCCESS: Embeddings file loaded successfully!")
except Exception as e:
    print("ERROR loading joblib:", e)

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"status": "Server is running smoothly!"}

@app.post("/api/chat")
def chat_endpoint(req: QueryRequest):
    user_query = req.query
    print(f"\nIncoming User Query: {user_query}")
    
    try:
        # Step A: Query Embedding Generation
        if hasattr(proces_incoming, 'create_embeddings_batch'):
            query_embedding = proces_incoming.create_embeddings_batch([user_query])[0]
        elif hasattr(proces_incoming, 'get_embedding'):
            query_embedding = proces_incoming.get_embedding(user_query)
        else:
            raise AttributeError("Embedding function missing in proces_incoming.py")

        q_emb = np.array(query_embedding).reshape(1, -1)
        
        # Step B: Vector Similarity Search
        scores = np.dot(doc_embeddings, q_emb.T).flatten()
        top_indices = np.argsort(scores)[-3:][::-1]
        matched_chunks = df.iloc[top_indices]
        
        # Step C: Construct Context & Sources
        context_text = ""
        sources = []
        for index, row in matched_chunks.iterrows():
            context_text += f"\n{row['text']}"
            sources.append({
                "title": row.get('title', 'Java Tutorial'),
                "timestamp": f"{row.get('start', '00:00')} - {row.get('end', '00:00')}",
                "text": str(row['text'])[:120] + "..."
            })
        
        # Step D: Prompt Generation & Ollama LLM Call
        prompt = f"Context from Java Video Tutorials:\n{context_text}\n\nUser Question: {user_query}\n\nProvide a clear and concise answer based ONLY on the context provided above:"
        
        if hasattr(proces_incoming, 'inference'):
            llm_out = proces_incoming.inference(prompt, model="llama3.2")
            response_text = llm_out.get("response", str(llm_out)) if isinstance(llm_out, dict) else str(llm_out)
        else:
            response_text = "Inference function not found in proces_incoming.py"
        
        return {
            "response": response_text,
            "sources": sources
        }

    except Exception as err:
        print("Backend Processing Error:", err)
        return {
            "response": f"Backend Processing Error: {str(err)}",
            "sources": []
        }