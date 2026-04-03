from fastapi import FastAPI
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

app = FastAPI()

# Load embeddings and Chroma DB once at startup
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(embedding_function=embeddings, persist_directory="chroma")

# Request model
class QueryRequest(BaseModel):
    question: str

# Health check — Azure uses this to confirm app is running
@app.get("/")
def health_check():
    return {"status": "RAG app is running!"}

# Query endpoint
@app.post("/query")
def query(request: QueryRequest):
    results = db.similarity_search_with_relevance_scores(request.question, k=5)

    if not results:
        return {"answer": "No relevant information found."}

    context = "\n\n--\n\n".join([doc.page_content for doc, score in results])

    return {
        "question": request.question,
        "context": context
    }