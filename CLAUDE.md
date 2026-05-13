# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**MediAid AI** — a medical assistant that combines a FAISS-backed RAG system over CDC/WHO documents, OCR for prescriptions/lab reports, and two scikit-learn risk-prediction pipelines (heart disease, diabetes). Users interact through a Streamlit web app or a FastAPI service.

## Commands

```bash
# First-time setup
pip install -r requirements.txt
cp .env.example .env                       # then edit and add OPENAI_API_KEY
python scripts/build_faiss_index.py        # builds rag_data/medical_embeddings.index from rag_data/embedded/*.json

# Run the app
streamlit run streamlit_app.py             # main UI (auth, search, OCR, risk-check)
python app.py                              # FastAPI keyword-search service (uses utils/faiss_utils.FAISSVectorStore)

# Tests / smoke checks (no pytest harness — these are runnable scripts)
python tests/test_rag_simple.py            # validates OPENAI key, FAISS index, end-to-end RAG query
python tests/test_rag_system.py            # broader RAG checks
python test_guardrail.py                   # is_medical_query() classifier sanity check

# Demos
python demos/medical_rag_demo.py
python demos/simple_rag_demo.py
python demos/demo_search.py

# Rebuild ML models from notebooks
jupyter notebook heartattack.ipynb         # writes models/heart_pipeline.pkl
jupyter notebook diabetes.ipynb            # writes models/diabetes_pipeline.pkl
```

Note: README references `streamlit_app2.py`, but only `streamlit_app.py` exists in this repo. Treat `streamlit_app.py` as the canonical entry point.

## Architecture

The system has three roughly independent capabilities glued together by the Streamlit app:

### 1. RAG pipeline (CDC + WHO knowledge base)
- **Data flow:** `data/*.json` (raw scrapes) → `scripts/cdcclean.py` / `whoclean.py` → `cleaned/*_cleaned.json` → `scripts/chunk_texts.py` → `rag_data/{cdc,who}_chunks.json` → `scripts/embed_chunks.py` (OpenAI embeddings) → `rag_data/embedded/{cdc,who}_embeddings.json` → `scripts/build_faiss_index.py` → `rag_data/medical_embeddings.index` + `medical_embeddings_metadata.json`.
- **Runtime:** `utils/faiss_utils.FAISSVectorStore` (cosine via `IndexFlatIP`) is loaded once and reused. `utils/rag.MedicalRAG` wraps it: embed query → FAISS search → format retrieved docs into a system prompt → `gpt-3.5-turbo` chat completion. Embedding model defaults to `text-embedding-ada-002` in `utils/rag.py` but `streamlit_app.py` configures LlamaIndex with `text-embedding-3-small` — these are separate code paths and the dimension match only holds because the FAISS index was built with whatever model `embed_chunks.py` used. If you regenerate embeddings, keep the model consistent across `embed_chunks.py`, `utils/rag.py`, and `streamlit_app.py`.
- **LlamaIndex path (in `streamlit_app.py` only):** `load_llamaindex_search()` builds a small in-memory `VectorStoreIndex` from the first 100 entries of each embedding file — this is a separate, lighter-weight retrieval path used by parts of the UI, not a replacement for the FAISS index.

### 2. OCR / document analysis
- `utils/ocr_utils.MedicalOCR` prefers Tesseract (binary path resolved via `utils/platform_utils.get_tesseract_default_paths()` per OS, or `TESSERACT_CMD` env var). EasyOCR is in `requirements.txt` as a fallback but the code intentionally avoids it ("avoid torch/EasyOCR native crashes").
- PDFs with embedded text are parsed via `pdfplumber` (`extract_text_from_pdf_bytes`); image-based PDFs go through `pdf2image` → Tesseract.

### 3. Risk-prediction models
- `models/heart_pipeline.pkl` and `models/diabetes_pipeline.pkl` are scikit-learn `Pipeline` objects (preprocessor + classifier) trained in the root-level notebooks. `streamlit_app.predict_with_pipeline()` calls `predict_proba` (or falls back to `decision_function` / `predict`) so the pipelines must expose one of those. Training data lives in `datasets/`.
- After a risk score, the UI feeds the result back into the RAG path for follow-up Q&A — `last_risk_context` in session state is the bridge.

### Import path quirk
Modules inside `utils/` import each other by bare name (e.g. `utils/rag.py` does `from faiss_utils import FAISSVectorStore`, not `from utils.faiss_utils import ...`). Callers compensate by appending `utils/` to `sys.path` before importing (`streamlit_app.py`, `app.py`, `tests/test_rag_simple.py` all do this). Preserve this pattern when adding new files in `utils/` or expect imports to break.

### Safety guardrail
`is_medical_query()` (defined in `streamlit_app.py` and mirrored in `test_guardrail.py`) is a keyword/pattern classifier that gates the RAG entry point. If you change the keyword lists in one place, update the other.

## Configuration

- `OPENAI_API_KEY` is required. Resolution order in `streamlit_app.py._load_api_key()`: `.env` next to the script → `.env` in CWD → `st.secrets`. `utils/rag.py` and `app.py` use plain `dotenv.load_dotenv()` from CWD only.
- Optional env overrides used by `utils/rag.py`: `OPENAI_EMBEDDING_MODEL`, `OPENAI_CHAT_MODEL`, `OPENAI_TEMPERATURE`, `TESSERACT_CMD`.
- `streamlit_app.py` pins BLAS thread counts to 1 (`OMP_NUM_THREADS`, etc.) at import time to avoid native crashes when FAISS, sklearn, and torch are loaded together — keep this if you reorganize imports.
