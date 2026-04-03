FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir \
    langchain \
    langchain-community \
    langchain-huggingface \
    langchain-text-splitters \
    sentence-transformers \
    chromadb \
    pypdf \
    python-dotenv \
    fastapi \
    uvicorn

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]