# query_data_rag_online.py
# Fully functional RAG with public Hugging Face model
# ---------------------------------------------------
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# -----------------------------
# Config
# -----------------------------
chroma_path = "chroma"  # Path to your persisted Chroma DB
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
hf_model_name = "tiiuae/falcon-7b-instruct"  # Public HF model
top_k = 5
max_tokens = 250

# -----------------------------
# Load embeddings and Chroma DB
# -----------------------------
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
db = Chroma(embedding_function=embeddings, persist_directory=chroma_path)

# -----------------------------
# Get user query
# -----------------------------
query_text = input("Enter your query: ").strip()
if not query_text:
    print("❌ Query cannot be empty.")
    exit()

# -----------------------------
# Retrieve top-k chunks
# -----------------------------
results = db.similarity_search_with_relevance_scores(query_text, k=top_k)
if not results:
    print(f"❌ No matching results found for: {query_text}")
    exit()

print("\n🔹 Retrieved Chunks:")
context_text = ""
for i, (doc, score) in enumerate(results, 1):
    print(f"\n[{i}] Score: {score:.4f}\n{doc.page_content}\n{'-'*60}")
    context_text += doc.page_content + "\n\n--\n\n"

# -----------------------------
# Load Hugging Face online model
# -----------------------------
print("\n⏳ Loading Hugging Face model from hub (this may take a while)...")
tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
model = AutoModelForCausalLM.from_pretrained(hf_model_name)
llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# -----------------------------
# Generate answer
# -----------------------------
# This is the prompt which is given to the llm
prompt = f"Answer the following question based on the context below.\n\nContext:\n{context_text}\nQuestion: {query_text}\nAnswer:"

resp = llm_pipeline(prompt, max_new_tokens=max_tokens, do_sample=True, temperature=0.7)
answer = resp[0]["generated_text"]

print("\n✅ Generated Answer:\n")
print(answer)
