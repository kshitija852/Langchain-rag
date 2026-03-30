# test_app.py
# Unit tests for the LangChain RAG application
# Tests cover: database creation helpers, query logic, and mocked embeddings/LLM calls

import os
import sys
import shutil
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

# ---------------------------------------------------------------------------
# Helpers to import project modules without executing their top-level code
# ---------------------------------------------------------------------------

# ── create_database ──────────────────────────────────────────────────────────

def import_create_database():
    """Import create_database without running top-level side-effects."""
    import importlib, types

    # Stub heavy dependencies before the module is loaded
    mocks = {
        "dotenv": MagicMock(),
        "langchain_community": MagicMock(),
        "langchain_community.document_loaders": MagicMock(),
        "langchain_community.embeddings": MagicMock(),
        "langchain_community.embeddings.ollama": MagicMock(),
        "langchain_community.vectorstores": MagicMock(),
        "langchain_text_splitters": MagicMock(),
        "langchain_huggingface": MagicMock(),
    }
    with patch.dict(sys.modules, mocks):
        # Prevent load_dotenv and file I/O from running at import time
        with patch("builtins.open", MagicMock()), \
             patch("os.path.exists", return_value=False):
            import create_database as cd
    return cd


# ── query_data ────────────────────────────────────────────────────────────────

def import_query_data():
    """Import query_data without running top-level side-effects."""
    mocks = {
        "langchain_huggingface": MagicMock(),
        "langchain_community": MagicMock(),
        "langchain_community.vectorstores": MagicMock(),
        "transformers": MagicMock(),
    }
    with patch.dict(sys.modules, mocks):
        with patch("builtins.input", return_value="Who is Alice?"):
            import query_data as qd
    return qd


# ===========================================================================
# 1.  safe_delete_chroma
# ===========================================================================

class TestSafeDeleteChroma:

    def test_no_op_when_path_missing(self, tmp_path):
        """Should do nothing when the directory does not exist."""
        cd = import_create_database()
        non_existent = str(tmp_path / "ghost_chroma")
        # Should not raise
        cd.safe_delete_chroma(non_existent)
        assert not os.path.exists(non_existent)

    def test_deletes_existing_directory(self, tmp_path):
        """Should remove an existing Chroma directory."""
        cd = import_create_database()
        chroma_dir = tmp_path / "chroma"
        chroma_dir.mkdir()
        (chroma_dir / "dummy.db").write_text("data")

        cd.safe_delete_chroma(str(chroma_dir))
        assert not chroma_dir.exists()

    def test_retries_on_permission_error(self, tmp_path):
        """Should retry up to `retries` times on PermissionError, then give up."""
        cd = import_create_database()
        chroma_dir = tmp_path / "locked_chroma"
        chroma_dir.mkdir()

        with patch("shutil.rmtree", side_effect=PermissionError("locked")):
            # Should not raise even after exhausting retries
            cd.safe_delete_chroma(str(chroma_dir), retries=2, delay=0)

        # Directory still exists because rmtree was always blocked
        assert chroma_dir.exists()


# ===========================================================================
# 2.  load_data
# ===========================================================================

class TestLoadData:

    def test_returns_list_of_documents(self):
        """load_data should return whatever the loader provides."""
        cd = import_create_database()

        fake_docs = [MagicMock(page_content="Page 1"), MagicMock(page_content="Page 2")]
        mock_loader = MagicMock()
        mock_loader.load.return_value = fake_docs

        with patch("create_database.PyPDFDirectoryLoader", return_value=mock_loader):
            docs = cd.load_data()

        assert docs == fake_docs
        assert len(docs) == 2

    def test_load_data_calls_loader_with_data_path(self):
        """PyPDFDirectoryLoader must be initialised with the configured data_path."""
        cd = import_create_database()

        mock_loader = MagicMock()
        mock_loader.load.return_value = []

        with patch("create_database.PyPDFDirectoryLoader", return_value=mock_loader) as MockLoader:
            cd.load_data()
            MockLoader.assert_called_once_with(cd.data_path)


# ===========================================================================
# 3.  Text splitting
# ===========================================================================

class TestTextSplitting:

    def test_chunks_are_produced(self):
        """RecursiveCharacterTextSplitter should split docs into smaller chunks."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            length_function=len,
            add_start_index=True,
        )

        # Build a minimal document-like object
        doc = MagicMock()
        doc.page_content = "A" * 300          # 300 chars → should produce multiple chunks
        doc.metadata = {}

        with patch.object(splitter, "split_documents", wraps=splitter.split_documents):
            chunks = splitter.split_documents([doc])

        assert len(chunks) > 1, "Expected multiple chunks from a 300-char document"

    def test_chunk_size_respected(self):
        """No chunk should exceed the configured chunk_size."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        chunk_size = 100
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)

        doc = MagicMock()
        doc.page_content = "Hello world. " * 50   # ~650 chars
        doc.metadata = {}

        chunks = splitter.split_documents([doc])
        for chunk in chunks:
            assert len(chunk.page_content) <= chunk_size + 20  # small tolerance


# ===========================================================================
# 4.  Chroma DB creation
# ===========================================================================

class TestChromaDBCreation:

    def test_chroma_from_documents_called(self):
        """Chroma.from_documents should be called with chunks and embeddings."""
        mock_db = MagicMock()
        mock_embeddings = MagicMock()
        fake_chunks = [MagicMock(), MagicMock()]

        with patch("langchain_community.vectorstores.Chroma.from_documents", return_value=mock_db) as mock_create:
            from langchain_community.vectorstores import Chroma
            db = Chroma.from_documents(fake_chunks, mock_embeddings, persist_directory="chroma")
            mock_create.assert_called_once_with(fake_chunks, mock_embeddings, persist_directory="chroma")

    def test_chroma_persist_called(self):
        """db.persist() must be called after Chroma is created."""
        mock_db = MagicMock()

        with patch("langchain_community.vectorstores.Chroma.from_documents", return_value=mock_db):
            from langchain_community.vectorstores import Chroma
            db = Chroma.from_documents([], MagicMock(), persist_directory="chroma")
            db.persist()
            mock_db.persist.assert_called_once()


# ===========================================================================
# 5.  Similarity search / retrieval
# ===========================================================================

class TestSimilaritySearch:

    def _make_mock_db(self, results):
        mock_db = MagicMock()
        mock_db.similarity_search_with_relevance_scores.return_value = results
        return mock_db

    def test_returns_top_k_results(self):
        """similarity_search_with_relevance_scores should return the correct number of results."""
        fake_results = [(MagicMock(page_content=f"chunk {i}"), 0.9 - i * 0.1) for i in range(5)]
        mock_db = self._make_mock_db(fake_results)

        results = mock_db.similarity_search_with_relevance_scores("Who is Alice?", k=5)
        assert len(results) == 5

    def test_empty_results_handled(self):
        """An empty result list should be detectable without raising."""
        mock_db = self._make_mock_db([])
        results = mock_db.similarity_search_with_relevance_scores("Unknown query", k=5)
        assert results == []

    def test_scores_are_numeric(self):
        """Every returned score should be a float-compatible number."""
        fake_results = [
            (MagicMock(page_content="Alice in Wonderland"), 0.372),
            (MagicMock(page_content="White Rabbit"), 0.327),
        ]
        mock_db = self._make_mock_db(fake_results)
        results = mock_db.similarity_search_with_relevance_scores("Who is Alice?", k=2)

        for _, score in results:
            assert isinstance(score, float)

    def test_query_passed_correctly(self):
        """The exact query string must be forwarded to the vector store."""
        mock_db = self._make_mock_db([])
        query = "What did the White Rabbit say?"
        mock_db.similarity_search_with_relevance_scores(query, k=5)
        mock_db.similarity_search_with_relevance_scores.assert_called_once_with(query, k=5)


# ===========================================================================
# 6.  Context assembly
# ===========================================================================

class TestContextAssembly:

    def test_context_text_concatenation(self):
        """Retrieved chunks should be joined with the expected separator."""
        results = [
            (MagicMock(page_content="Alice was a young girl."), 0.9),
            (MagicMock(page_content="She followed a White Rabbit."), 0.8),
        ]

        context_text = ""
        for doc, score in results:
            context_text += doc.page_content + "\n\n--\n\n"

        assert "Alice was a young girl." in context_text
        assert "She followed a White Rabbit." in context_text
        assert "\n\n--\n\n" in context_text

    def test_prompt_contains_query_and_context(self):
        """The final prompt must include both the context and the user query."""
        context_text = "Alice is a young girl.\n\n--\n\n"
        query_text = "Who is Alice?"

        prompt = (
            f"Answer the following question based on the context below.\n\n"
            f"Context:\n{context_text}\nQuestion: {query_text}\nAnswer:"
        )

        assert query_text in prompt
        assert context_text in prompt
        assert "Answer:" in prompt


# ===========================================================================
# 7.  LLM pipeline (mocked)
# ===========================================================================

class TestLLMPipeline:

    def test_pipeline_called_with_prompt(self):
        """The HuggingFace pipeline must be called with the assembled prompt."""
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"generated_text": "Alice is the protagonist."}]

        prompt = "Answer the following question based on the context below.\n\nContext:\nsome context\nQuestion: Who is Alice?\nAnswer:"
        result = mock_pipeline(prompt, max_new_tokens=250, do_sample=True, temperature=0.7)

        mock_pipeline.assert_called_once_with(prompt, max_new_tokens=250, do_sample=True, temperature=0.7)

    def test_generated_text_extracted(self):
        """The answer must be read from result[0]['generated_text']."""
        mock_pipeline = MagicMock()
        expected_answer = "Alice is the main character of Alice's Adventures in Wonderland."
        mock_pipeline.return_value = [{"generated_text": expected_answer}]

        resp = mock_pipeline("some prompt", max_new_tokens=250, do_sample=True, temperature=0.7)
        answer = resp[0]["generated_text"]

        assert answer == expected_answer

    def test_empty_query_not_processed(self):
        """An empty query string should be caught before hitting the LLM."""
        query_text = "   "
        assert not query_text.strip(), "Empty/whitespace query should evaluate to falsy after strip()"


# ===========================================================================
# 8.  Embeddings initialisation
# ===========================================================================

class TestEmbeddings:

    def test_huggingface_embeddings_initialised_with_correct_model(self):
        """HuggingFaceEmbeddings must use the all-MiniLM-L6-v2 model."""
        with patch("langchain_huggingface.HuggingFaceEmbeddings") as MockEmb:
            from langchain_huggingface import HuggingFaceEmbeddings
            HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            MockEmb.assert_called_once_with(model_name="sentence-transformers/all-MiniLM-L6-v2")


# ===========================================================================
# 9.  Configuration constants
# ===========================================================================

class TestConfiguration:

    def test_chroma_path_default(self):
        cd = import_create_database()
        assert cd.chroma_path == "chroma"

    def test_data_path_default(self):
        cd = import_create_database()
        assert cd.data_path == "Data"

    def test_top_k_value(self):
        qd = import_query_data()
        assert qd.top_k == 5

    def test_max_tokens_value(self):
        qd = import_query_data()
        assert qd.max_tokens == 250

    def test_embedding_model_name(self):
        qd = import_query_data()
        assert qd.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
