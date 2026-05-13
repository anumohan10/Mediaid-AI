#!/usr/bin/env python3
"""
MediAid AI - Interactive Medical Search Interface
Streamlit web app for searching medical information from CDC and WHO databases
"""

import streamlit as st
import os
import sys
import re
import json
import traceback
from datetime import datetime
from typing import List, Dict, Optional
from dotenv import load_dotenv
from pathlib import Path
from llama_index.core import Settings, VectorStoreIndex, Document
from llama_index.llms.openai import OpenAI as LIOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import joblib
import streamlit as st
from auth.auth_ui import render_auth_page
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# OpenAI embedding model used for query embeddings AND for building the FAISS
# index. text-embedding-3-small replaces the older text-embedding-ada-002:
# it is newer, ~5x cheaper, and outperforms ada-002 on MTEB benchmarks.
# Keep this constant in sync with scripts/embed_chunks.py and utils/rag.py;
# if you change it, regenerate the FAISS index — query vectors from a
# different model will not match documents already embedded with this one.
EMBEDDING_MODEL = "text-embedding-3-small"

# Streamlit page config — must be the FIRST Streamlit command in the script,
# so it has to run before _load_api_key() (which may call st.warning).
st.set_page_config(
    page_title="MediAid AI - Medical Search",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---- API key loading (robust) ----------------------------------------------
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
import os

def big_emoji(emoji: str = "🏥", size_px: int = 96):
    st.markdown(
        f"""
        <div style="text-align:center;">
            <span style="font-size:{size_px}px; line-height:1">{emoji}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
import time

def render_onboarding_prelogin():
    # Hide sidebar while onboarding (optional)
    st.markdown("""
        <style>
        [data-testid="stSidebar"] { display: none; }
        </style>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        big_emoji("🏥", size_px=140)
        st.markdown(
            "<h1 style='text-align:center;margin:0'>MediAid AI</h1>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<p style='text-align:center;font-size:18px;color:#666'>"
            "Your evidence-based medical information companion"
            "</p>", unsafe_allow_html=True
        )

        with st.spinner("Preparing your workspace…"):
            prog = st.progress(0)
            for i in range(0, 101, 8):
                time.sleep(0.06)
                prog.progress(i)

        # mark onboarding complete for this session, then go to login
        st.session_state.did_onboard = True
        st.rerun()


def _load_api_key() -> str | None:
    # 1) .env next to this file
    script_env = Path(__file__).parent / ".env"
    if script_env.exists():
        load_dotenv(dotenv_path=script_env, override=False)

    # 2) .env in current working directory (useful when launching via `streamlit run`)
    cwd_env = Path.cwd() / ".env"
    if cwd_env.exists():
        load_dotenv(dotenv_path=cwd_env, override=False)

    # 3) streamlit secrets (if set)
    key = os.getenv("OPENAI_API_KEY")
    if (not key) and hasattr(st, "secrets"):
        try:
            key = st.secrets.get("OPENAI_API_KEY", None)
        except Exception:
            pass

    # normalize empty/placeholder
    if key and key.strip().lower() != "your-api-key-here":
        return key.strip()

    # No usable key — surface a clear, actionable warning in the UI.
    st.warning(
        "**OpenAI API key not found.** "
        "AI summaries, chat, and document analysis will be disabled.\n\n"
        "**To enable them:**\n"
        "1. Copy `.env.example` to `.env` in the project root.\n"
        "2. Add the line `OPENAI_API_KEY=sk-...` with your real key.\n"
        "3. Restart the Streamlit app (`streamlit run streamlit_app.py`).\n\n"
        "Alternatively, set `OPENAI_API_KEY` in `.streamlit/secrets.toml` "
        "or export it as an environment variable before launching."
    )
    return None

OPENAI_API_KEY = _load_api_key()

# Minimal, safe debug (shows only prefix/suffix so you don't leak it)
def _mask(k: str) -> str:
    return f"{k[:6]}…{k[-4:]}" if k and len(k) > 12 else str(bool(k))

print("API Key Loaded?", bool(OPENAI_API_KEY))
if OPENAI_API_KEY:
    print("API key (masked):", _mask(OPENAI_API_KEY))

# ---- Configure LlamaIndex only if key exists -------------------------------
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI as LIOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

if OPENAI_API_KEY:
    Settings.llm = LIOpenAI(model="gpt-3.5-turbo", temperature=0.1, api_key=OPENAI_API_KEY)
    Settings.embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
else:
    # Do NOT initialize LlamaIndex or embeddings when the key is missing
    Settings.llm = None
    Settings.embed_model = None

# Add utils to path so utils/*.py modules can import each other by bare name
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'
if 'selected_disease' not in st.session_state:
    st.session_state.selected_disease = None
if 'search_query' not in st.session_state:
    st.session_state.search_query = ""
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'chat_context' not in st.session_state:
    st.session_state.chat_context = []
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'risk_chat_history' not in st.session_state:
    st.session_state.risk_chat_history = []
if 'last_risk_context' not in st.session_state:
    st.session_state.last_risk_context = ""

@st.cache_resource
def load_medical_search():
    """Load the FAISS-backed medical document store.

    Returns the loaded vector store, or None if the index is missing/corrupt —
    in which case a clear, actionable error is rendered in the app so the
    operator knows exactly how to rebuild it.
    """
    index_path = "rag_data/medical_embeddings.index"

    if not os.path.exists(index_path):
        st.error(
            f"**FAISS index not found at `{index_path}`.**\n\n"
            "The medical database cannot be loaded until the index is built.\n\n"
            "**To build the index:**\n"
            "1. Ensure `OPENAI_API_KEY` is set in `.env`.\n"
            "2. Run `python scripts/build_faiss_index.py` from the project root.\n"
            "   This reads `rag_data/embedded/*.json` and writes the index "
            "plus `rag_data/medical_embeddings_metadata.json`.\n"
            "3. Restart the Streamlit app."
        )
        return None

    try:
        from utils.faiss_utils import FAISSVectorStore

        vector_store = FAISSVectorStore()
        vector_store.load_index(index_path)
        return vector_store
    except Exception as e:
        st.error(
            f"**Failed to load FAISS index from `{index_path}`:** {e}\n\n"
            "The file may be corrupt or incompatible with the current "
            "`faiss-cpu` version. Try rebuilding it with "
            "`python scripts/build_faiss_index.py`."
        )
        return None

@st.cache_resource
def load_llamaindex_search():
    """Load the optional LlamaIndex query engine.

    Returns the engine on success, or None if LlamaIndex is unavailable for
    any reason (missing package, no API key, no embedding files, network
    error, etc.). All failures are silent in the UI — diagnostics go to the
    server log only — so an absent LlamaIndex backend never disrupts the
    user-facing experience. The caller checks for None and degrades to the
    FAISS-only path.
    """
    try:
        import os
        import json

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your-api-key-here":
            print("[load_llamaindex_search] skipped: OPENAI_API_KEY not set")
            return None

        from llama_index.core import VectorStoreIndex, Document, Settings
        from llama_index.llms.openai import OpenAI as LIOpenAI
        from llama_index.embeddings.openai import OpenAIEmbedding

        Settings.llm = LIOpenAI(model="gpt-3.5-turbo", temperature=0.1)
        Settings.embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL)

        documents = []
        who_path = "rag_data/embedded/who_embeddings.json"
        if os.path.exists(who_path):
            with open(who_path, "r", encoding="utf-8") as f:
                for item in json.load(f)[:100]:
                    documents.append(Document(
                        text=item.get("text", ""),
                        metadata={"source": "WHO", **item.get("metadata", {})},
                    ))
        cdc_path = "rag_data/embedded/cdc_embeddings.json"
        if os.path.exists(cdc_path):
            with open(cdc_path, "r", encoding="utf-8") as f:
                for item in json.load(f)[:100]:
                    documents.append(Document(
                        text=item.get("text", ""),
                        metadata={"source": "CDC", **item.get("metadata", {})},
                    ))

        if not documents:
            print("[load_llamaindex_search] skipped: no embedding files found")
            return None

        index = VectorStoreIndex.from_documents(documents)
        return index.as_query_engine(
            similarity_top_k=5,
            response_mode="tree_summarize",
            verbose=False,
        )
    except Exception as e:
        # Silent failure — log to server, return None so callers can fall back.
        print(f"[load_llamaindex_search] disabled: {e}")
        return None


@st.cache_data
def get_medical_topics(_vector_store):
    """Get list of available medical topics"""
    if not _vector_store:
        return []
    
    topics = set()
    for meta in _vector_store.metadata:
        text = meta.get('text', '')
        if 'Disease:' in text:
            topic = text.split('Disease:')[1].split('\n')[0].strip()
            if topic and len(topic) > 0:
                topics.add(topic)
    
    return sorted(list(topics))

@st.cache_resource
def load_models():
    """Load heart & diabetes risk pipelines from the models folder."""
    import os, joblib

    models = {}
    paths = {
        "heart": "models/heart_pipeline.pkl",
        "diabetes": "models/diabetes_pipeline.pkl",
    }

    for key, path in paths.items():
        try:
            if os.path.exists(path):
                models[key] = joblib.load(path, mmap_mode=None)
                # st.info(f"Loaded {key} model from {os.path.basename(path)}")
            else:
                st.warning(f"{key.capitalize()} model file not found: {path}")
        except Exception as e:
            st.error(f"Could not load {key} model: {e}")

    return models

def format_percent(p):
    try:
        return f"{float(p)*100:.1f}%"
    except Exception:
        return str(p)

def predict_with_pipeline(model, X):
    """Return (risk_pct, yhat) from a sklearn pipeline or estimator."""
    # proba of positive class if available, else fallback
    if hasattr(model, "predict_proba"):
        risk = float(model.predict_proba(X)[0][1])
    elif hasattr(model, "decision_function"):
        import numpy as np
        z = float(model.decision_function(X)[0])
        risk = 1.0 / (1.0 + np.exp(-z))   # logistic squash
    else:
        risk = float(model.predict(X)[0])  # 0/1
    yhat = int(model.predict(X)[0])
    return risk, yhat

def search_documents_llamaindex(llamaindex_engine, query: str) -> str:
    """Search using LlamaIndex query engine with enhanced prompting"""
    if not llamaindex_engine:
        return None
    
    try:
        # Enhanced query with detailed prompting for comprehensive medical responses
        enhanced_query = f"""
        Based on the medical information available, please provide a comprehensive answer about: "{query}"

        Please structure your response as follows:
        1. **Overview**: What is this condition/topic?
        2. **Key Facts**: Important characteristics, symptoms, or features
        3. **Prevention/Treatment**: Available prevention methods or treatments (if applicable)
        4. **When to Seek Help**: When to consult healthcare professionals
        5. **Additional Notes**: Any other important information

        Use bullet points where appropriate and ensure the response is medically accurate.
        Always emphasize consulting healthcare professionals for specific medical advice.
        
        Question: {query}
        """
        
        # Query using LlamaIndex query engine with enhanced prompt
        response = llamaindex_engine.query(enhanced_query)
        
        # Add medical disclaimer to LlamaIndex responses
        enhanced_response = f"{str(response)}\n\nImportant: Always consult healthcare professionals for specific medical advice."
        
        return enhanced_response
        
    except Exception as e:
        st.error(f"LlamaIndex search error: {e}")
        return None

def search_documents(vector_store, query: str, max_results: int = 10) -> List[Dict]:
    """Search for medical documents.

    Returns an empty list on any failure (and prints a warning to the server
    log) so callers can degrade gracefully rather than crashing the page.
    """
    if not vector_store:
        return []

    try:
        # Parse query into keywords
        query_words = re.findall(r'\b\w+\b', query.lower())
        query_words = [w for w in query_words if len(w) > 2]

        if not query_words:
            return []

        # Build simple keyword index if not exists
        if not hasattr(vector_store, 'keyword_index'):
            keyword_index = {}
            for i, text in enumerate(vector_store.texts):
                words = re.findall(r'\b\w+\b', text.lower())
                for word in words:
                    if len(word) > 2:
                        if word not in keyword_index:
                            keyword_index[word] = []
                        keyword_index[word].append(i)
            vector_store.keyword_index = keyword_index

        # Find matching documents
        doc_scores = {}
        for word in query_words:
            if word in vector_store.keyword_index:
                for doc_idx in vector_store.keyword_index[word]:
                    if doc_idx not in doc_scores:
                        doc_scores[doc_idx] = 0

                    text = vector_store.texts[doc_idx].lower()
                    word_count = text.count(word)

                    # Boost score for title/heading matches
                    if word in text[:100]:
                        word_count *= 2

                    doc_scores[doc_idx] += word_count

        # Sort by relevance
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # Format results
        results = []
        for doc_idx, score in sorted_docs[:max_results]:
            result = {
                'text': vector_store.texts[doc_idx],
                'metadata': vector_store.metadata[doc_idx],
                'relevance_score': score,
                'source': vector_store.metadata[doc_idx].get('source', 'Unknown'),
                'index': doc_idx,
            }
            results.append(result)

        return results
    except Exception as e:
        print(f"[search_documents] failed for query={query!r}: {e}")
        traceback.print_exc()
        return []

def extract_medical_entities(text: str) -> Optional[Dict[str, List[str]]]:
    """Extract structured medical entities from a document via OpenAI.

    Returns a dict with keys ``medications``, ``lab_values``, ``diagnoses``,
    ``vitals`` (each a list of strings — possibly empty), or ``None`` if the
    API key is missing or the call/parse fails. The caller renders a
    fallback UI ("AI analysis not available") when this returns ``None``.

    The model is asked to return JSON only; we strip code fences and slice
    to the outermost braces before parsing to defend against the usual
    stray-prose failure mode.
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or api_key == 'your-api-key-here':
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        snippet = (text or "")[:2000]
        prompt = (
            "Extract medical information from this document text.\n"
            "Return ONLY a JSON object with no extra text:\n"
            "{\n"
            "  'medications': ['list of medication/agent names only'],\n"
            "  'lab_values': ['list of values with numbers e.g. SUVmax 5.4'],\n"
            "  'diagnoses': ['list of conditions/diagnoses found'],\n"
            "  'vitals': ['list of vital signs if present']\n"
            "}\n"
            "If nothing found in a category return empty list.\n"
            f"Document: {snippet}"
        )

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=500,
        )
        raw = (response.choices[0].message.content or "").strip()

        # Strip ```json ... ``` fences if present, then slice to the
        # outermost braces so trailing prose doesn't break json.loads.
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        payload = raw[start:end + 1]
        # The prompt uses single quotes; coerce to JSON-valid double quotes.
        payload = payload.replace("'", '"')
        data = json.loads(payload)

        # Normalize to the four expected keys with list values.
        result: Dict[str, List[str]] = {}
        for key in ('medications', 'lab_values', 'diagnoses', 'vitals'):
            value = data.get(key, [])
            if isinstance(value, list):
                result[key] = [str(v).strip() for v in value if str(v).strip()]
            else:
                result[key] = []
        return result
    except Exception as e:
        print(f"[extract_medical_entities] failed: {e}")
        return None


def get_openai_summary(query: str, results: List[Dict]) -> Optional[str]:
    """Generate AI summary using OpenAI"""
    try:
        from openai import OpenAI
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key or api_key == 'your-api-key-here':
            return None
        
        client = OpenAI(api_key=api_key)
        
        # Prepare context from top results
        context_parts = []
        for result in results[:3]:
            source = result['source']
            text = result['text'][:800]  # Increased text length for better context
            context_parts.append(f"Source: {source}\n{text}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Create prompt
        prompt = f"""Based on the following medical information, provide a comprehensive answer about: "{query}"

Medical Information:
{context}

Please provide:
1. **Overview**: What is this condition/topic?
2. **Key Facts**: Important characteristics, symptoms, or features
3. **Prevention/Treatment**: Available prevention methods or treatments (if applicable)
4. **When to Seek Help**: When to consult healthcare professionals
5. **Additional Notes**: Any other important information

Format your response with clear headings and bullet points where appropriate.
Always emphasize consulting healthcare professionals for specific medical advice."""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful medical AI assistant. Provide accurate, evidence-based information while always recommending professional medical consultation. Format your response clearly with headings and bullet points."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1200,
            temperature=0.1
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"OpenAI Error: {e}")
        return None

def get_keyword_summary(query: str, results: List[Dict]) -> str:
    """Generate simple summary without OpenAI"""
    if not results:
        return "No relevant information found in the medical database."
    
    sources = set(result['source'] for result in results)
    
    summary = f"**Found {len(results)} relevant documents** from {', '.join(sources)}.\n\n"
    
    # Extract key information from top results
    key_points = []
    for result in results[:5]:  # Use top 5 results
        text = result['text']
        # Extract first sentence or key information
        sentences = text.split('. ')
        if sentences:
            key_point = sentences[0][:250]
            if not key_point.endswith('.'):
                key_point += "..."
            key_points.append(f"• **{result['source']}**: {key_point}")
    
    if key_points:
        summary += "**Key Information:**\n" + "\n\n".join(key_points)
    
    summary += "\n\nImportant: Always consult healthcare professionals for specific medical advice."
    
    return summary

def is_medical_query(query: str) -> bool:
    """Check if the query is medical-related and appropriate for MediAid AI"""
    query_lower = query.lower()
    
    # Medical keywords that indicate legitimate medical queries
    medical_keywords = [
        # Medical conditions
        'symptom', 'symptoms', 'disease', 'condition', 'syndrome', 'disorder', 'infection',
        'diabetes', 'hypertension', 'cancer', 'heart', 'kidney', 'liver', 'lung', 'brain',
        'asthma', 'allergy', 'allergies', 'pneumonia', 'flu', 'fever', 'cough', 'pain',
        'headache', 'migraine', 'depression', 'anxiety', 'stroke', 'blood pressure',
        
        # Medical terms
        'treatment', 'therapy', 'medication', 'medicine', 'drug', 'prescription',
        'surgery', 'operation', 'procedure', 'diagnosis', 'test', 'screening',
        'vaccine', 'vaccination', 'immunization', 'prevention', 'cure',
        
        # Body parts and systems
        'chest', 'abdomen', 'stomach', 'throat', 'eye', 'ear', 'nose', 'mouth',
        'skin', 'bone', 'muscle', 'joint', 'blood', 'urine', 'bowel', 'bladder',
        
        # Medical professionals and settings
        'doctor', 'physician', 'nurse', 'hospital', 'clinic', 'emergency',
        'medical', 'health', 'healthcare', 'patient', 'diagnosis', 'prognosis',
        
        # Pregnancy and child health
        'pregnant', 'pregnancy', 'baby', 'infant', 'child', 'pediatric',
        'maternal', 'fetal', 'prenatal', 'postnatal',
        
        # Common medical questions
        'what is', 'how to treat', 'side effects', 'safe during', 'contraindication',
        'interaction', 'dosage', 'dose', 'risk', 'complication'
    ]
    
    # Non-medical topics that should be blocked
    non_medical_keywords = [
        # Technology
        'computer', 'software', 'programming', 'code', 'coding', 'website', 'app',
        'internet', 'wifi', 'smartphone', 'iphone', 'android', 'windows', 'mac',
        
        # Entertainment
        'movie', 'film', 'tv show', 'music', 'song', 'game', 'gaming', 'sport',
        'football', 'basketball', 'soccer', 'celebrity', 'actor', 'actress',
        
        # Business/Finance
        'business', 'investment', 'stock', 'money', 'bank', 'loan', 'credit',
        'insurance', 'tax', 'salary', 'job', 'career', 'resume',
        
        # Education (non-medical)
        'homework', 'essay', 'assignment', 'math', 'physics', 'chemistry', 'history',
        'geography', 'literature', 'university', 'college', 'degree',
        
        # Politics/Social
        'politics', 'government', 'election', 'president', 'vote', 'law', 'legal',
        'court', 'judge', 'lawyer', 'attorney',
        
        # Travel/Food (non-medical)
        'vacation', 'travel', 'hotel', 'restaurant', 'recipe', 'cooking', 'food',
        'cuisine', 'drink', 'alcohol', 'wine', 'beer',
        
        # Personal/Relationships
        'relationship', 'dating', 'love', 'marriage', 'divorce', 'family',
        'friend', 'friendship', 'social media', 'facebook', 'twitter', 'instagram'
    ]
    
    # Check for obvious non-medical content
    non_medical_count = sum(1 for keyword in non_medical_keywords if keyword in query_lower)
    medical_count = sum(1 for keyword in medical_keywords if keyword in query_lower)
    
    # If query contains significantly more non-medical terms, likely not medical
    if non_medical_count > medical_count and non_medical_count >= 2:
        return False
    
    # Check for explicit non-medical patterns
    non_medical_patterns = [
        'how to make', 'recipe for', 'best restaurant', 'movie recommendation',
        'song lyrics', 'game walkthrough', 'investment advice', 'stock price',
        'political opinion', 'latest news', 'weather forecast', 'sports score',
        'programming tutorial', 'code example', 'homework help', 'essay writing'
    ]
    
    if any(pattern in query_lower for pattern in non_medical_patterns):
        return False
    
    # If query contains medical keywords or medical question patterns, allow it
    medical_question_patterns = [
        'what is', 'what are', 'how to treat', 'how to prevent', 'symptoms of',
        'causes of', 'treatment for', 'medication for', 'safe during', 'side effects'
    ]
    
    has_medical_pattern = any(pattern in query_lower for pattern in medical_question_patterns)
    has_medical_keyword = medical_count > 0
    
    # Allow if it has medical context or seems like a health question
    if has_medical_keyword or has_medical_pattern:
        return True
    
    # For very short queries, be more lenient (might be medical abbreviations)
    if len(query.split()) <= 3:
        # But block obvious non-medical short phrases
        obvious_non_medical = ['pizza', 'movie', 'game', 'music', 'weather', 'news', 'politics', 'programming', 'tutorial']
        if any(term in query_lower for term in obvious_non_medical):
            return False
        return True
    
    # Default: if we can't clearly identify it as medical, be cautious
    return False

def is_complex_query(query: str) -> bool:
    """Determine if a query requires task decomposition"""
    complexity_indicators = [
        'multiple conditions', 'and', 'with', 'also have', 'plus', 'additionally',
        'interactions', 'safe during', 'pregnant', 'elderly', 'child',
        'taking medication', 'side effects', 'contraindications'
    ]
    
    query_lower = query.lower()
    # Count conditions mentioned
    medical_conditions = ['diabetes', 'hypertension', 'heart', 'kidney', 'liver', 'pregnancy', 'cancer', 'asthma']
    condition_count = len([word for word in medical_conditions if word in query_lower])
    
    # Check for complexity indicators
    has_complexity_words = any(indicator in query_lower for indicator in complexity_indicators)
    
    # Consider complex if multiple conditions OR complexity indicators present
    return condition_count > 1 or has_complexity_words or len(query.split()) > 15

def decompose_medical_query(query: str, client) -> Optional[Dict]:
    """Break down complex medical queries into sub-tasks"""
    try:
        decomposition_prompt = f"""
        Analyze this medical query and break it down into logical sub-tasks: "{query}"
        
        Return a JSON structure with:
        {{
            "main_topic": "Brief description of the main question",
            "conditions": ["condition1", "condition2"],
            "sub_questions": ["specific question 1", "specific question 2"],
            "safety_considerations": ["safety aspect 1", "safety aspect 2"],
            "complexity_level": "low/medium/high"
        }}
        
        Only return valid JSON, no other text.
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": decomposition_prompt}],
            max_tokens=400,
            temperature=0.1
        )
        
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Query decomposition failed: {e}")
        return None

def execute_complex_search(decomposition: Dict, vector_store) -> Dict:
    """Execute searches for each sub-task of a complex query"""
    results = {
        "condition_searches": {},
        "safety_searches": {},
        "question_searches": {}
    }
    
    # Search for each condition
    for condition in decomposition.get("conditions", []):
        if condition.strip():
            condition_results = search_documents(vector_store, f"{condition} symptoms treatment management", max_results=3)
            results["condition_searches"][condition] = condition_results
    
    # Search for safety considerations
    for safety in decomposition.get("safety_considerations", []):
        if safety.strip():
            safety_results = search_documents(vector_store, safety, max_results=2)
            results["safety_searches"][safety] = safety_results
    
    # Search for specific sub-questions
    for question in decomposition.get("sub_questions", []):
        if question.strip():
            question_results = search_documents(vector_store, question, max_results=3)
            results["question_searches"][question] = question_results
    
    return results

def synthesize_complex_response(user_message: str, decomposition: Dict, complex_results: Dict, chat_history: List[Dict], client) -> str:
    """Synthesize comprehensive response from multiple search results"""
    try:
        # Format all search results
        formatted_results = "RESEARCH FINDINGS:\n\n"
        
        # Add condition information
        if complex_results["condition_searches"]:
            formatted_results += "CONDITIONS RESEARCHED:\n"
            for condition, results in complex_results["condition_searches"].items():
                formatted_results += f"\n{condition.upper()}:\n"
                for i, result in enumerate(results[:2], 1):
                    formatted_results += f"{i}. {result['text'][:200]}...\n"
        
        # Add safety information
        if complex_results["safety_searches"]:
            formatted_results += "\nSAFETY CONSIDERATIONS:\n"
            for safety, results in complex_results["safety_searches"].items():
                formatted_results += f"\n{safety}:\n"
                for i, result in enumerate(results[:1], 1):
                    formatted_results += f"{i}. {result['text'][:200]}...\n"
        
        # Add specific question answers
        if complex_results["question_searches"]:
            formatted_results += "\nSPECIFIC QUESTIONS:\n"
            for question, results in complex_results["question_searches"].items():
                formatted_results += f"\nQ: {question}\n"
                if results:
                    formatted_results += f"A: {results[0]['text'][:200]}...\n"
        
        # Build conversation context
        conversation_context = ""
        if chat_history:
            recent_messages = chat_history[-3:]  # Last 3 exchanges
            for msg in recent_messages:
                conversation_context += f"User: {msg['user']}\nAssistant: {msg['assistant'][:100]}...\n\n"
        
        synthesis_prompt = f"""
        You are MediAid AI providing comprehensive medical information analysis.
        
        ORIGINAL COMPLEX QUERY: "{user_message}"
        
        CONVERSATION HISTORY:
        {conversation_context}
        
        DECOMPOSED RESEARCH:
        Main Topic: {decomposition.get('main_topic', 'N/A')}
        Conditions: {', '.join(decomposition.get('conditions', []))}
        Complexity: {decomposition.get('complexity_level', 'medium')}
        
        {formatted_results}
        
        Provide a comprehensive, structured response that:
        
        1. **OVERVIEW**: Summarize the main medical question
        2. **CONDITION ANALYSIS**: Address each condition mentioned with key facts
        3. **SAFETY CONSIDERATIONS**: Highlight important safety information, interactions, contraindications
        4. **SPECIFIC RECOMMENDATIONS**: Answer each sub-question clearly
        5. **ACTION PLAN**: Provide clear next steps
        6. **MEDICAL DISCLAIMER**: Emphasize professional consultation
        
        Format with clear headings, bullet points, and easy-to-understand language.
        Be thorough but concise. Focus on evidence-based information.
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": synthesis_prompt}],
            max_tokens=1000,
            temperature=0.2
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"Response synthesis failed: {e}")
        return "I apologize, but I encountered an error while analyzing your complex query. Please try rephrasing your question or ask about specific conditions separately."

def get_conversational_response(user_message: str, chat_history: List[Dict], results: List[Dict], document_context: str = "") -> str:
    """Generate conversational response with context from chat history and search results"""
    try:
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key or api_key == 'your-api-key-here':
            return None
            
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        # Build context from search results
        context_parts = []
        for result in results[:3]:
            source = result['source']
            text = result['text'][:600]
            context_parts.append(f"Source: {source}\n{text}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Build conversation history
        conversation_context = ""
        if chat_history:
            recent_messages = chat_history[-6:]  # Last 6 messages for context
            for msg in recent_messages:
                conversation_context += f"User: {msg['user']}\nAssistant: {msg['assistant']}\n\n"
        
        # Create conversational prompt
        prompt = f"""You are MediAid AI, a helpful medical information assistant. You provide accurate, easy-to-understand medical information based on CDC and WHO data.

Previous conversation:
{conversation_context}

Current medical information available:
{context}"""

        # Add document context if available
        if document_context:
            prompt += f"""

USER'S UPLOADED MEDICAL DOCUMENT:
{document_context}

IMPORTANT: The user has uploaded a medical document (shown above). Please analyze this document and answer their question based on both the uploaded document content and the general medical information provided. Explain any medical terms or values found in their document in simple language."""

        prompt += f"""

User's current question: "{user_message}"

Please respond in a natural, conversational way. Be helpful, empathetic, and provide clear medical information. If this is a follow-up question, reference the previous conversation appropriately. {"If the user uploaded a document, focus on explaining their specific document while providing relevant general medical context." if document_context else ""} Always end with a reminder to consult healthcare professionals for personal medical advice.

Your response should be:
- Conversational and natural
- Medically accurate based on the provided sources{"and uploaded document" if document_context else ""}
- Easy to understand
- Appropriately empathetic
- Include relevant follow-up suggestions if appropriate"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.3
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"AI Response Error: {e}")
        return None

def ask_followup_with_rag(vector_store, question: str, extra_context: str, llamaindex_engine=None):
    """
    If the LlamaIndex toggle is ON and an engine exists, answer with LlamaIndex (no FAISS sources).
    Otherwise: search FAISS with the user's question, then answer with OpenAI using both the FAISS
    snippets and the extra_context (risk/document summary).
    """
    use_llama = bool(st.session_state.get("use_llamaindex")) and (llamaindex_engine is not None)

    if use_llama:
        # LlamaIndex path (no FAISS sources)
        augmented = (
            f"USER CONTEXT (risk/report summary):\n{extra_context}\n\n"
            f"USER QUESTION:\n{question}\n\n"
            "Give a concise, safe, medically grounded answer."
        )
        reply = search_documents_llamaindex(llamaindex_engine, augmented)
        return reply, []  # no sources (LlamaIndex handled retrieval)
    else:
        # FAISS path: search with the *question* (not the big augmented block)
        results = search_documents(vector_store, question, max_results=8)

        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        use_ai = api_key and api_key != 'your-api-key-here' and len(api_key) > 20

        if use_ai:
            reply = get_conversational_response(
                user_message=question,
                chat_history=st.session_state.risk_chat_history,
                results=results,
                document_context=extra_context,  # <- feed risk/report summary as context
            )
            return reply, results
        else:
            return get_keyword_summary(question, results), results



def render_navigation():
    """Render navigation sidebar"""
    st.sidebar.markdown(
        """
        <div style="display:flex;align-items:center;gap:12px;margin:0 0 16px 0;">
            <div style="width:40px;height:40px;background:#1D4ED8;border-radius:10px;
                        display:flex;align-items:center;justify-content:center;
                        flex-shrink:0;">
                <svg width="22" height="22" viewBox="0 0 22 22" fill="none"
                     xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                    <path d="M11 4v14M4 11h14" stroke="#FFFFFF" stroke-width="2.5"
                          stroke-linecap="round"/>
                </svg>
            </div>
            <div style="font-size:18px;font-weight:600;color:#0F172A;
                        font-family:'Inter',sans-serif;line-height:1;">
                MediAid AI
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _current = st.session_state.current_page

    def _nav_type(page_key: str) -> str:
        return "primary" if _current == page_key else "secondary"

    # Navigation buttons
    if st.sidebar.button("Home", use_container_width=True, key="nav_home",
                         icon=":material/home:", type=_nav_type("home")):
        st.session_state.current_page = 'home'
        st.session_state.selected_disease = None
        st.rerun()

    if st.sidebar.button("Search", use_container_width=True, key="nav_search",
                         icon=":material/search:", type=_nav_type("search")):
        st.session_state.current_page = 'search'
        st.session_state.selected_disease = None
        st.rerun()

    if st.sidebar.button("Upload & Ask", use_container_width=True, key="nav_upload",
                         icon=":material/upload:", type=_nav_type("upload")):
        st.session_state.current_page = 'upload'
        st.session_state.selected_disease = None
        st.rerun()

    if st.sidebar.button("Risk Check", use_container_width=True, key="nav_risk",
                         icon=":material/monitor_heart:", type=_nav_type("risk")):
        st.session_state.current_page = 'risk'
        st.session_state.selected_disease = None
        st.rerun()

    if st.sidebar.button("Browse Topics", use_container_width=True, key="nav_browse",
                         icon=":material/menu_book:",
                         type=_nav_type("browse") if _current != "disease_detail" else "primary"):
        st.session_state.current_page = 'browse'
        st.session_state.selected_disease = None
        st.rerun()

    if st.sidebar.button("FAQs", use_container_width=True, key="nav_faqs",
                         icon=":material/help:", type=_nav_type("faqs")):
        st.session_state.current_page = 'faqs'
        st.session_state.selected_disease = None
        st.rerun()

    if st.sidebar.button("Search History", use_container_width=True, key="nav_history",
                         icon=":material/history:", type=_nav_type("history")):
        st.session_state.current_page = 'history'
        st.session_state.selected_disease = None
        st.rerun()
    
    # Show current disease if selected
    if st.session_state.selected_disease:
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**Current Topic:**")
        st.sidebar.info(st.session_state.selected_disease)
        
        if st.sidebar.button("⬅️ Back to Browse", use_container_width=True, key="nav_back_browse"):
            st.session_state.current_page = 'browse'
            st.session_state.selected_disease = None
            st.rerun()
    
    # Check OpenAI status
    st.sidebar.markdown("---")
    
    # Try to load environment variables again
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    # Show LlamaIndex status
    if 'llamaindex_engine' in st.session_state and st.session_state.llamaindex_engine:
        # st.sidebar.success("🦙 LlamaIndex: Ready")
        use_llamaindex = st.sidebar.checkbox("Detailed responses", value=False, key="use_llamaindex")
        compare_engines = st.sidebar.checkbox("Compare search engines", value=False, key="compare_engines")
    else:
        st.sidebar.warning("🦙 LlamaIndex: Not Available")
        use_llamaindex = False
        compare_engines = False
    
    # Removed debug information for production
    
    if api_key and api_key != 'your-api-key-here' and len(api_key) > 20:
        # st.sidebar.success("🤖 AI Summaries: Enabled")
        use_ai = st.sidebar.checkbox("AI summaries", value=True, key="ai_summaries_diabetes")
        
        # # Task Decomposition Feature
        # st.sidebar.success("🧠 Task Decomposition: Enabled")
        # st.sidebar.info("Complex queries will be automatically analyzed and broken down into sub-tasks")
        
        # # Content Guardrails
        # st.sidebar.success("🔒 Content Guardrails: Active")
        # st.sidebar.info("Non-medical queries are automatically blocked")
        
        # Removed test button for production
    else:
        st.sidebar.warning("🤖 AI Summaries: Disabled (API key not configured)")
        st.sidebar.warning("🧠 Task Decomposition: Disabled (requires OpenAI API)")
        st.sidebar.success("🔒 Content Guardrails: Active")
        use_ai = False
    
    # Logout button at the bottom
    st.sidebar.markdown("---")
    _username = st.session_state.username or ""
    _initial = (_username[:1] or "?").upper()
    st.sidebar.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:10px;margin:8px 0 12px 0;">
            <div style="width:32px;height:32px;border-radius:50%;background:#1D4ED8;
                        color:#FFFFFF;display:flex;align-items:center;justify-content:center;
                        font-weight:600;font-size:14px;font-family:'Inter',sans-serif;">
                {_initial}
            </div>
            <div style="line-height:1.2;font-family:'Inter',sans-serif;">
                <div style="color:#94A3B8;font-size:11px;text-transform:uppercase;
                            letter-spacing:0.04em;">Logged in as</div>
                <div style="color:#0F172A;font-size:14px;font-weight:500;">{_username}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Policy reminder
    st.sidebar.markdown("---")
    st.sidebar.info("🔒 **Policy**: MediAid AI is designed exclusively for medical and health-related questions. Non-medical queries will be blocked.")
    
    if st.sidebar.button("Logout", use_container_width=True, key="nav_logout", icon=":material/logout:"):
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.session_state.current_page = 'home'
        st.rerun()

def render_home_page(vector_store):
    """Render the home page"""
    st.markdown(
        "<h1 class='no-title-dot'>MediAid AI — Medical Information Search</h1>",
        unsafe_allow_html=True,
    )
    st.markdown("**Your comprehensive medical information assistant powered by CDC and WHO databases**")
    
    # Show system status
    col1, col2 = st.columns(2)
    
    # with col1:
    #     if vector_store:
    #         st.success(f"✅ FAISS: Loaded {vector_store.index.ntotal} medical documents")
    #     else:
    #         st.error("❌ FAISS: Not loaded")
    
    # with col2:
    #     llamaindex_query_engine = st.session_state.get('llamaindex_engine', None)
    #     if llamaindex_query_engine:
    #         st.success("✅ LlamaIndex: Ready for intelligent search")
    #     else:
    #         st.warning("⚠️ LlamaIndex: Not available (requires OpenAI API)")
    
    # # Quick stats
    # col1, col2, col3, col4 = st.columns(4)
    # with col1:
    #     st.metric("Total Documents", "5,489")
    # with col2:
    #     st.metric("Medical Topics", "1,068+")
    # with col3:
    #     st.metric("Data Sources", "CDC + WHO")
    # with col4:
    #     st.metric("Search Types", "FAISS + LlamaIndex")
    
    # st.markdown("---")
    
    # Features overview
    st.subheader("Features")
    
    col1, col2 = st.columns(2)
    
    _card_style = (
        "background:#FFFFFF;"
        "border:1px solid #E2E8F0;"
        "border-radius:12px;"
        "padding:20px;"
        "box-shadow:0 1px 3px rgba(15,23,42,0.04);"
        "margin-bottom:16px;"
    )
    _heading_style = (
        "color:#0F172A;font-size:18px;font-weight:500;"
        "margin:0 0 12px 0;font-family:'Inter',sans-serif;"
    )
    _list_style = (
        "color:#475569;font-size:15px;line-height:1.7;"
        "margin:0;padding-left:20px;"
    )

    with col1:
        st.markdown(
            f"""
            <div style="{_card_style}">
                <h3 style="{_heading_style}">Smart Search</h3>
                <ul style="{_list_style}">
                    <li>Fast keyword-based search</li>
                    <li>Intelligent semantic search</li>
                    <li>AI-powered summaries</li>
                    <li>Source attribution</li>
                </ul>
            </div>
            <div style="{_card_style}">
                <h3 style="{_heading_style}">Browse Topics</h3>
                <ul style="{_list_style}">
                    <li>1,068+ medical conditions</li>
                    <li>Dedicated disease pages</li>
                    <li>Detailed information</li>
                    <li>CDC and WHO data</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div style="{_card_style}">
                <h3 style="{_heading_style}">Safety &amp; Compliance</h3>
                <ul style="{_list_style}">
                    <li><strong>Content Guardrails:</strong> Medical-only queries enforced</li>
                    <li><strong>Policy Protection:</strong> Non-medical content blocked</li>
                    <li><strong>User Authentication:</strong> Secure access control</li>
                    <li><strong>Search History:</strong> Personal activity tracking</li>
                </ul>
            </div>
            <div style="{_card_style}">
                <h3 style="{_heading_style}">Key Benefits</h3>
                <ul style="{_list_style}">
                    <li>Evidence-based information</li>
                    <li>Multiple search engines</li>
                    <li>Mobile responsive</li>
                    <li>Always up-to-date</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    # # Search engine comparison
    # if llamaindex_query_engine:
    #     st.markdown("---")
    #     st.subheader("🔍 Search Engine Comparison")
        
    #     comparison_col1, comparison_col2 = st.columns(2)
        
    #     with comparison_col1:
    #         st.markdown("""
    #         **🏃‍♂️ FAISS Search:**
    #         - ⚡ Ultra-fast keyword matching
    #         - 📊 Relevance scoring
    #         - 🎯 Exact term matching
    #         - 💾 Low memory usage
    #         """)
        
    #     with comparison_col2:
    #         st.markdown("""
    #         **🦙 LlamaIndex Search:**
    #         - 🧠 Understands context and meaning
    #         - 💬 Natural language queries
    #         - 📖 Comprehensive responses
    #         - 🤖 AI-powered insights
    #         """)
    
    # # Task Decomposition Feature Demo
    # st.markdown("---")
    # st.subheader("🧠 Advanced Task Decomposition")
    # st.markdown("**New Agentic AI Feature:** Automatically breaks down complex medical queries into manageable sub-tasks")
    
    # col1, col2 = st.columns(2)
    
    # with col1:
    #     st.markdown("""
    #     **🔍 What It Does:**
    #     - Analyzes complex multi-condition queries
    #     - Breaks them into specific research tasks
    #     - Searches each aspect systematically
    #     - Synthesizes comprehensive responses
    #     """)
        
    #     st.markdown("""
    #     **🎯 Perfect For:**
    #     - Multiple medical conditions
    #     - Drug interactions & safety
    #     - Pregnancy-related questions
    #     - Elderly care considerations
    #     """)
    
    # with col2:
    #     st.markdown("""
    #     **💡 Example Complex Queries:**
    #     - *"I have diabetes and high blood pressure, am pregnant, what medications are safe?"*
    #     - *"Elderly patient with heart disease and kidney problems - drug interactions?"*
    #     - *"Child with asthma and allergies - vaccine safety considerations?"*
    #     """)
        
        # Demo button
        if st.button("Try Demo", key="complex_demo"):
            demo_query = "I have diabetes and high blood pressure and I'm pregnant. What medications are safe?"
            st.session_state.search_query = demo_query
            st.session_state.current_page = 'search'
            st.rerun()
    
def render_search_page(vector_store):
    """Chat-style medical search page built on Streamlit's native chat components."""
    st.title("Chat with MediAid AI")
    st.markdown(
        "Ask me anything about medical conditions, symptoms, treatments, "
        "or prevention. I can help with follow-up questions too!"
    )

    # AI status (sidebar)
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    use_ai = bool(api_key) and api_key != 'your-api-key-here' and len(api_key) > 20
    if use_ai:
        st.sidebar.success("AI Chat: Enabled")
    else:
        st.sidebar.warning("AI Chat: Disabled (using keyword responses)")

    use_llamaindex = st.session_state.get('use_llamaindex', False)
    compare_engines = st.session_state.get('compare_engines', False)
    llamaindex_engine = st.session_state.get('llamaindex_engine', None)

    # A click on a welcome suggestion sets pending_question and reruns;
    # we consume it here so it flows through the normal message pipeline.
    pending_question = st.session_state.pop("pending_question", None)

    # Replay prior conversation using native chat_message bubbles.
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(chat['user'])
        with st.chat_message("assistant"):
            st.markdown(chat['assistant'])

    # Welcome state — only when there is genuinely nothing happening yet.
    if not st.session_state.chat_history and not pending_question:
        st.markdown(
            "<div class='welcome-anchor'></div>"
            "<div style='margin:24px 0 12px 0;color:#475569;font-size:14px;'>"
            "Try one of these to get started:"
            "</div>",
            unsafe_allow_html=True,
        )
        suggestions = [
            "What are symptoms of diabetes?",
            "How to prevent heart disease?",
            "What is hypertension?",
        ]
        cols = st.columns(3)
        for col, question in zip(cols, suggestions):
            with col:
                if st.button(question, key=f"welcome_{question}",
                             use_container_width=True):
                    st.session_state.pending_question = question
                    st.rerun()

    # Clear-chat control (only shows once there's history).
    if st.session_state.chat_history:
        col_clear, col_count = st.columns([1, 4])
        with col_clear:
            if st.button("Clear chat", key="clear_chat_btn"):
                st.session_state.chat_history = []
                st.session_state.chat_context = []
                st.rerun()
        with col_count:
            st.caption(f"{len(st.session_state.chat_history)} messages in chat")

    # Chat input — automatically pinned to the bottom of the main view.
    typed_message = st.chat_input("Type your medical question here...")
    user_message = pending_question or typed_message
    if not user_message:
        return

    # Render the user's message immediately, before any model calls.
    with st.chat_message("user"):
        st.markdown(user_message)

    # Guardrail: block non-medical queries with an assistant-side reply.
    if not is_medical_query(user_message):
        with st.chat_message("assistant"):
            st.error(
                "**Policy violation.** MediAid AI is designed exclusively for "
                "medical and health-related questions. Please ask about medical "
                "conditions, symptoms, treatments, medications, or health concerns."
            )
            st.markdown(
                "**Examples of appropriate questions:**\n"
                "- What are the symptoms of diabetes?\n"
                "- How to treat high blood pressure?\n"
                "- Side effects of aspirin\n"
                "- Safe medications during pregnancy"
            )
        return

    # --- Comparison view: render both engines side-by-side, don't persist ---
    if compare_engines and llamaindex_engine:
        with st.chat_message("assistant"):
            st.markdown("**Comparing search engines**")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("### Enhanced AI response")
                with st.spinner("Thinking..."):
                    llamaindex_response = search_documents_llamaindex(
                        llamaindex_engine, user_message
                    )
                if llamaindex_response:
                    st.markdown(llamaindex_response)
                else:
                    st.error("LlamaIndex search failed.")
            with col_b:
                st.markdown("### FAISS + AI response")
                with st.spinner("Thinking..."):
                    results = search_documents(vector_store, user_message, max_results=8)
                if results and use_ai:
                    ai_response = get_conversational_response(
                        user_message, st.session_state.chat_history, results
                    )
                    if ai_response:
                        st.markdown(ai_response)
                    else:
                        st.error("AI response failed.")
                elif results:
                    st.markdown(get_keyword_summary(user_message, results))
                else:
                    st.error("No results found.")
        return

    # --- LlamaIndex-only path ---
    if use_llamaindex and llamaindex_engine:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                llamaindex_response = search_documents_llamaindex(
                    llamaindex_engine, user_message
                )
            if llamaindex_response:
                reply = f"**Enhanced response:**\n\n{llamaindex_response}"
                st.markdown(reply)
                st.session_state.chat_history.append(
                    {'user': user_message, 'assistant': reply}
                )
                save_search_history(
                    st.session_state.username, user_message,
                    llamaindex_response, "llamaindex_search",
                )
                return
            st.warning("LlamaIndex search failed. Falling back to standard search.")
            # Fall through into the FAISS path below, inside the same bubble.
            results = search_documents(vector_store, user_message, max_results=8)
            _render_faiss_reply(results, user_message, use_ai, source_tag="ai_search")
        return

    # --- Standard FAISS / complex-query path (single assistant bubble) ---
    with st.chat_message("assistant"):
        is_complex = is_complex_query(user_message)

        if is_complex and use_ai:
            with st.spinner("Thinking..."):
                from openai import OpenAI
                client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                decomposition = decompose_medical_query(user_message, client)

            if decomposition:
                with st.expander("Query analysis", expanded=False):
                    st.markdown(f"**Main topic:** {decomposition.get('main_topic', 'N/A')}")
                    st.markdown(
                        f"**Conditions to research:** "
                        f"{', '.join(decomposition.get('conditions', []))}"
                    )
                    st.markdown(
                        f"**Complexity level:** "
                        f"{decomposition.get('complexity_level', 'medium').title()}"
                    )
                    if decomposition.get('sub_questions'):
                        st.markdown("**Sub-questions to address:**")
                        for q in decomposition['sub_questions']:
                            st.write(f"• {q}")

                with st.spinner("Thinking..."):
                    complex_results = execute_complex_search(decomposition, vector_store)
                    complex_response = synthesize_complex_response(
                        user_message, decomposition, complex_results,
                        st.session_state.chat_history, client,
                    )

                if complex_response:
                    st.markdown(complex_response)
                    with st.expander("Research sources"):
                        for category, label in (
                            ("condition_searches", "Condition research"),
                            ("safety_searches", "Safety research"),
                        ):
                            bucket = complex_results.get(category, {})
                            if not bucket:
                                continue
                            st.markdown(f"**{label}**")
                            for key, res in bucket.items():
                                st.markdown(f"*{key}:*")
                                for i, r in enumerate(res[:2], 1):
                                    st.write(
                                        f"{i}. {r['source']} — {r['text'][:150]}..."
                                    )
                    st.session_state.chat_history.append(
                        {'user': user_message, 'assistant': complex_response}
                    )
                    save_search_history(
                        st.session_state.username, user_message,
                        complex_response, "complex_analysis",
                    )
                    return
                st.warning("Complex analysis failed. Falling back to standard search.")

            # Decomposition failed or synthesis failed — fall back to FAISS.
            with st.spinner("Thinking..."):
                results = search_documents(vector_store, user_message, max_results=8)
        else:
            with st.spinner("Thinking..."):
                results = search_documents(vector_store, user_message, max_results=8)

        _render_faiss_reply(results, user_message, use_ai, source_tag="ai_search")


def _render_faiss_reply(results, user_message, use_ai, source_tag):
    """Render a FAISS+AI (or keyword-fallback) reply inside the current chat bubble."""
    if not results:
        st.warning(
            "I couldn't find relevant information for your question. "
            "Please try rephrasing or ask about common medical topics."
        )
        return

    if use_ai:
        with st.spinner("Thinking..."):
            ai_response = get_conversational_response(
                user_message, st.session_state.chat_history, results
            )
        if ai_response:
            st.markdown(ai_response)
            with st.expander("Sources used"):
                for i, r in enumerate(results[:3], 1):
                    st.markdown(
                        f"**{i}. {r['source']}** (Relevance: {r['relevance_score']})"
                    )
                    st.markdown(r['text'][:300] + "...")
                    if i < 3:
                        st.markdown("---")
            st.session_state.chat_history.append(
                {'user': user_message, 'assistant': ai_response}
            )
            save_search_history(
                st.session_state.username, user_message, ai_response, source_tag
            )
            return
        st.error("AI response unavailable. Please try again.")
        return

    # Keyword-only fallback.
    summary = get_keyword_summary(user_message, results)
    st.markdown(summary)
    st.session_state.chat_history.append(
        {'user': user_message, 'assistant': summary}
    )
    save_search_history(
        st.session_state.username, user_message, summary, "keyword_search"
    )

def render_upload_page(vector_store):
    """Upload a medical document and chat about it using native chat components."""
    st.title("Upload & Ask")
    st.markdown(
        "Upload medical reports, prescriptions, or lab results and ask "
        "questions about them!"
    )

    # OCR utilities
    try:
        from utils.ocr_utils import create_ocr_interface
        ocr_available = True
    except ImportError:
        ocr_available = False

    # AI / LlamaIndex status
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    use_ai = bool(api_key) and api_key != 'your-api-key-here' and len(api_key) > 20
    llamaindex_engine = st.session_state.get('llamaindex_engine', None)

    if use_ai:
        st.sidebar.success("AI Analysis: Enabled")
    else:
        st.sidebar.warning("AI Analysis: Disabled (using keyword responses)")

    # Use the global "Detailed responses" toggle from render_navigation()
    # (st.session_state.use_llamaindex) — same source of truth as the
    # Search and Risk Check pages, so the toggle behaves consistently
    # across the whole app. No per-page checkbox here.
    use_llamaindex = bool(st.session_state.get("use_llamaindex")) and bool(llamaindex_engine)

    if ocr_available:
        st.sidebar.success("Document Reading: Enabled")
    else:
        st.sidebar.error("Document Reading: Disabled")
        st.error("Document upload feature requires additional packages.")
        st.info("To enable document reading, install: `pip install easyocr opencv-python Pillow`")
        st.code("pip install easyocr opencv-python Pillow", language="bash")
        return

    # Demo / compliance disclaimer — shown immediately above the file
    # uploader so users see it before selecting a document.
    st.warning(
        "**Demo notice:** Do not upload real patient documents. "
        "This demo processes uploaded files using AI services. "
        "For production use, ensure HIPAA compliance and data "
        "processing agreements are in place."
    )

    # OCR / upload interface
    extracted_text = create_ocr_interface()

    if extracted_text:
        st.session_state.document_context = extracted_text
        st.session_state.upload_page_document = True

        st.success("Document processed successfully.")

        # --- Quick Analysis (LLM-extracted entities) ---
        # We cache the result per-document so the API is only called once
        # per upload, not on every Streamlit rerun. The cache key is the
        # document text itself, so re-uploading the same doc reuses the
        # earlier extraction.
        st.subheader("Quick Analysis")
        if (
            st.session_state.get('ai_entities_for') != extracted_text
            or 'ai_entities' not in st.session_state
        ):
            with st.spinner("Extracting medical entities..."):
                st.session_state.ai_entities = extract_medical_entities(extracted_text)
                st.session_state.ai_entities_for = extracted_text
        ai_entities = st.session_state.ai_entities

        if ai_entities is None:
            st.info("AI analysis not available.")
        else:
            def _bulletize(items):
                items = [i for i in (items or []) if str(i).strip()]
                if not items:
                    return "_None detected_"
                return "\n".join(f"- {item}" for item in items)

            cols = st.columns(2)
            with cols[0]:
                st.markdown("**Medications Found:**")
                st.markdown(_bulletize(ai_entities.get('medications')))
                st.markdown("**Diagnoses Found:**")
                st.markdown(_bulletize(ai_entities.get('diagnoses')))
            with cols[1]:
                st.markdown("**Lab Values Found:**")
                st.markdown(_bulletize(ai_entities.get('lab_values')))
                st.markdown("**Vitals Found:**")
                st.markdown(_bulletize(ai_entities.get('vitals')))

    # When no document has been loaded yet, show the "What You Can Upload"
    # guide and stop here.
    if not (
        st.session_state.get('document_context')
        and st.session_state.get('upload_page_document')
    ):
        st.info(
            "Upload a medical document above to start asking questions about it."
        )
        st.subheader("What You Can Upload")
        guide_col1, guide_col2 = st.columns(2)
        with guide_col1:
            st.markdown(
                "**Medical Reports:**\n"
                "• Blood test results\n"
                "• X-ray reports\n"
                "• MRI/CT scan reports\n"
                "• Pathology reports"
            )
            st.markdown(
                "**Prescriptions:**\n"
                "• Medication lists\n"
                "• Prescription slips\n"
                "• Pharmacy labels"
            )
        with guide_col2:
            st.markdown(
                "**Clinical Documents:**\n"
                "• Doctor's notes\n"
                "• Discharge summaries\n"
                "• Consultation letters\n"
                "• Medical certificates"
            )
            st.markdown(
                "**Lab Results:**\n"
                "• Complete blood count\n"
                "• Lipid panels\n"
                "• Glucose tests\n"
                "• Hormone levels"
            )
        return

    # Chat section for document-specific questions
    st.markdown("---")
    st.subheader("Ask Questions About Your Document")

    with st.expander("View Uploaded Document Text"):
        st.text_area(
            "Extracted Text:", st.session_state.document_context,
            height=200, disabled=True,
        )

    # Clear document
    col_clear, _ = st.columns([1, 4])
    with col_clear:
        if st.button("Clear Document", key="clear_upload_doc"):
            for k in ('document_context', 'document_analysis',
                      'upload_page_document', 'upload_chat_history',
                      'upload_question', 'process_upload_question',
                      'ai_entities', 'ai_entities_for'):
                st.session_state.pop(k, None)
            st.rerun()

    # Init per-document chat history
    if 'upload_chat_history' not in st.session_state:
        st.session_state.upload_chat_history = []

    # Replay history with native chat bubbles
    for chat in st.session_state.upload_chat_history:
        with st.chat_message("user"):
            st.markdown(chat['user'])
        with st.chat_message("assistant"):
            st.markdown(chat['assistant'])

    # Pending question from a suggested-question click. The button handler
    # sets `upload_question` plus the `process_upload_question` flag, then
    # reruns; we consume both here and process the question immediately,
    # rendering user + assistant bubbles inline (no need to pre-fill the
    # chat input — Streamlit's chat_input does not support that).
    if st.session_state.get("process_upload_question"):
        question = st.session_state.upload_question
        st.session_state.process_upload_question = False
        st.session_state.pop('upload_question', None)
        _process_upload_message(
            question, vector_store, use_ai,
            use_llamaindex, llamaindex_engine,
        )

    # Chat input — pinned to the bottom of the main view.
    typed_message = st.chat_input(
        "Ask a question about your uploaded document..."
    )
    if typed_message:
        _process_upload_message(
            typed_message, vector_store, use_ai,
            use_llamaindex, llamaindex_engine,
        )


def _process_upload_message(
    user_message,
    vector_store,
    use_ai,
    use_llamaindex,
    llamaindex_engine,
):
    """Render guardrail / user / assistant bubbles for one document question.

    Shared between the suggested-question flag branch and the chat_input
    branch in render_upload_page so both flows behave identically.

    ``use_llamaindex`` mirrors the global "Detailed responses" toggle in the
    sidebar (st.session_state.use_llamaindex), keeping behavior consistent
    with render_search_page() and render_risk_page().
    """
    # Guardrail
    if not is_medical_query(user_message):
        with st.chat_message("user"):
            st.markdown(user_message)
        with st.chat_message("assistant"):
            st.error(
                "**Policy violation.** MediAid AI is designed exclusively for "
                "medical and health-related questions. Please ask questions "
                "about your medical document."
            )
            st.markdown(
                "**Examples of appropriate questions about your document:**\n"
                "- What medications are mentioned?\n"
                "- What are the lab results?\n"
                "- Explain this medical report\n"
                "- Are there any concerning values?"
            )
        return

    # Show the user message immediately, before any model calls.
    with st.chat_message("user"):
        st.markdown(user_message)

    document_context = st.session_state.document_context
    enhanced_query = (
        "User uploaded a medical document with the following content:\n\n"
        f"DOCUMENT CONTENT:\n{document_context}\n\n"
        f"USER QUESTION ABOUT THE DOCUMENT:\n{user_message}\n\n"
        "Please answer the user's question based on the uploaded document "
        "content and provide additional relevant medical information if helpful."
    )

    with st.chat_message("assistant"):
        response_text = None
        source_tag = None
        results = []

        # --- LlamaIndex path (if "Detailed responses" is on in the sidebar) ---
        if use_llamaindex and llamaindex_engine:
            llamaindex_prompt = (
                "Based on the uploaded medical document and medical knowledge "
                "base, please provide a comprehensive analysis.\n\n"
                f"UPLOADED DOCUMENT CONTENT:\n{document_context}\n\n"
                f"USER'S QUESTION:\n{user_message}\n\n"
                "Please provide:\n"
                "1. **Document Analysis**: What the uploaded document shows\n"
                "2. **Answer to Question**: Direct response to the user's question\n"
                "3. **Medical Context**: Additional relevant medical information\n"
                "4. **Recommendations**: Next steps or considerations\n\n"
                "Base your response on both the document content and medical knowledge."
            )
            with st.spinner("Thinking..."):
                try:
                    response_text = search_documents_llamaindex(
                        llamaindex_engine, llamaindex_prompt
                    )
                except Exception as e:
                    st.warning(f"LlamaIndex error: {e}. Falling back to FAISS.")
                    response_text = None
            if response_text:
                source_tag = "llamaindex_document_analysis"
                st.markdown(response_text)
                with st.expander("Document Context Used"):
                    st.text_area(
                        "Document Content:", document_context,
                        height=200, disabled=True,
                    )

        # --- FAISS + AI / keyword fallback ---
        if not response_text:
            with st.spinner("Thinking..."):
                results = search_documents(
                    vector_store, enhanced_query, max_results=5
                )

            if results and use_ai:
                with st.spinner("Thinking..."):
                    try:
                        response_text = get_conversational_response(
                            enhanced_query,
                            st.session_state.upload_chat_history,
                            results,
                            document_context,
                        )
                    except Exception as e:
                        st.error(f"AI analysis error: {e}")
                        response_text = None
                if response_text:
                    source_tag = "faiss_document_analysis"
                    st.markdown(response_text)
                    with st.expander("Sources Used"):
                        st.markdown("**Your Uploaded Document**")
                        snippet = document_context[:500]
                        if len(document_context) > 500:
                            snippet += "..."
                        st.markdown(snippet)
                        st.markdown("---")
                        st.markdown("**Medical Database References**")
                        for i, r in enumerate(results[:3], 1):
                            st.markdown(
                                f"**{i}. {r['source']}** "
                                f"(Relevance: {r['relevance_score']})"
                            )
                            st.markdown(r['text'][:300] + "...")
                            if i < 3:
                                st.markdown("---")
                else:
                    st.error("AI analysis failed. Please try again.")
            elif results:
                # Keyword-only fallback
                summary = get_keyword_summary(user_message, results)
                response_text = (
                    "Based on your uploaded document and medical database:\n\n"
                    f"{summary}"
                )
                source_tag = "keyword_document_analysis"
                st.markdown(response_text)
            else:
                st.warning(
                    "No relevant medical information found. "
                    "Please try rephrasing your question."
                )

        if response_text:
            # Engine badge — mirrors render_search_page() / render_risk_page().
            if (st.session_state.get("use_llamaindex")
                    and st.session_state.get("llamaindex_engine")):
                st.caption("Answered with Enhanced AI")
            else:
                st.caption("Answered with AI")

            st.session_state.upload_chat_history.append(
                {'user': user_message, 'assistant': response_text}
            )
            save_search_history(
                st.session_state.username, user_message,
                response_text, source_tag or "document_analysis",
            )


def _render_risk_followup(chat_key: str, vector_store) -> None:
    """Render the post-prediction follow-up Q&A for one risk tab.

    Order: gate on last_risk_context, render history bubbles first, then
    the chat_input. On submit we append the turn (with engine + sources)
    to history and st.rerun(), so the just-finished turn moves up into
    the history loop on the next render — fixing the previous "new bubble
    appears below the input" bug.
    """
    # Show the follow-up section only after a risk has actually been
    # predicted (the prediction handler is what populates last_risk_context).
    if not st.session_state.get("last_risk_context"):
        return

    st.markdown("---")
    st.subheader("Ask follow-up about your result")
    st.caption("Your risk summary will be used as context with CDC/WHO facts.")

    # 1) Render history first — every prior turn (including the one we
    #    just appended on the previous rerun) shows above the input.
    for chat in st.session_state.risk_chat_history:
        with st.chat_message("user"):
            st.markdown(chat["user"])
        with st.chat_message("assistant"):
            st.markdown(chat["assistant"])
            if chat.get("engine"):
                st.caption(f"Answered with {chat['engine']}")
            sources = chat.get("sources") or []
            if sources:
                with st.expander("Sources"):
                    for i, r in enumerate(sources[:3], 1):
                        st.write(
                            f"**{i}. {r.get('source', 'Unknown')}** "
                            f"(relevance {r.get('relevance_score')})"
                        )
                        snippet = r.get("text") or ""
                        st.write(
                            snippet[:300] + ("…" if len(snippet) > 300 else "")
                        )

    # Clear-chat control — only when there's history. Same layout/style as
    # the Search page's clear button. chat_key is unique per tab so the
    # button key is too.
    if st.session_state.risk_chat_history:
        col_clear, col_count = st.columns([1, 4])
        with col_clear:
            if st.button("Clear chat", key=f"clear_{chat_key}"):
                st.session_state.risk_chat_history = []
                st.rerun()
        with col_count:
            st.caption(
                f"{len(st.session_state.risk_chat_history)} messages in chat"
            )

    # 2) Input
    q = st.chat_input("Ask about your result...", key=chat_key)
    if not q:
        return

    # 3) Process and append
    with st.spinner("Thinking..."):
        answer, results = ask_followup_with_rag(
            vector_store,
            q,
            st.session_state.last_risk_context,
            llamaindex_engine=st.session_state.get("llamaindex_engine"),
        )

    if not answer:
        st.error("Sorry, I couldn't generate an answer. Please try rephrasing.")
        return

    engine = (
        "Enhanced AI"
        if (st.session_state.get("use_llamaindex")
            and st.session_state.get("llamaindex_engine"))
        else "AI"
    )
    st.session_state.risk_chat_history.append({
        "user": q,
        "assistant": answer,
        "engine": engine,
        "sources": results,
    })

    # 4) Rerun so the new turn lands inside the history loop above the input.
    st.rerun()


def render_risk_page(vector_store):
    """Risk prediction forms + post-result RAG/LlamaIndex Q&A using pipelines."""
    import streamlit as st

    # ---- ensure state keys exist ----
    if "risk_chat_history" not in st.session_state:
        st.session_state.risk_chat_history = []
    if "last_risk_context" not in st.session_state:
        st.session_state.last_risk_context = ""

    st.title("Risk Check")

    # ---- load saved sklearn pipelines ----
    models = load_models()
    have_heart = bool(models.get("heart"))
    have_diab  = bool(models.get("diabetes"))

    if not (have_heart or have_diab):
        st.error(
            "No saved models found. Please export your trained pipelines to "
            "`models/heart_pipeline.pkl` and/or `models/diabetes_pipeline.pkl`."
        )
        return

    tab_heart, tab_diab = st.tabs(["❤️ Heart Disease", "🧪 Diabetes"])

    # =========================
    #          HEART
    # =========================
    with tab_heart:
        if not have_heart:
            st.info("Heart model not loaded.")
        else:
            st.subheader("Heart Disease Risk")

            c1, c2, c3 = st.columns(3)
            with c1:
                age      = st.number_input("Age", min_value=0, max_value=120, value=55, step=1)
                chol     = st.number_input("Cholesterol (mg/dL)", min_value=50, max_value=500, value=240, step=1)
                trestbps = st.number_input("Resting BP (mm Hg)", min_value=60, max_value=260, value=130, step=1)
            with c2:
                thalach  = st.number_input("Max Heart Rate", min_value=40, max_value=240, value=150, step=1)
                oldpeak  = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
                exang    = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
            with c3:
                sex  = st.selectbox("Sex", ["Female", "Male"])
                fbs  = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
                cp   = st.selectbox("Chest Pain Type", ["typical", "atypical", "non-anginal", "asymptomatic"])

            col_a, col_b = st.columns(2)
            with col_a:
                restecg = st.selectbox(
                    "Resting ECG", ["normal", "ST-T wave abnormality", "left ventricular hypertrophy"]
                )
            with col_b:
                slope = st.selectbox("ST Segment Slope", ["upsloping", "flat", "downsloping"])

            # encoders (must match training)
            cp_map = {"typical": 0, "atypical": 1, "non-anginal": 2, "asymptomatic": 3}
            restecg_map = {"normal": 0, "ST-T wave abnormality": 1, "left ventricular hypertrophy": 2}
            slope_map = {"upsloping": 0, "flat": 1, "downsloping": 2}

            # 11-feature vector (UCI-like)
            X_heart = [[
                age,
                1 if sex == "Male" else 0,
                cp_map[cp],
                trestbps,
                chol,
                1 if fbs == "Yes" else 0,
                restecg_map[restecg],
                thalach,
                1 if exang == "Yes" else 0,
                oldpeak,
                slope_map[slope],
            ]]

            if st.button("Predict Heart Risk", type="primary", key="predict_heart"):
                try:
                    model = models["heart"]
                    risk_pct, _ = predict_with_pipeline(model, X_heart)
                    risk_text = f"**Heart Disease Risk:** {format_percent(risk_pct)}"
                    st.success(risk_text)

                    explain = [
                        f"- Age: {age}, Sex: {sex}",
                        f"- Cholesterol: {chol} mg/dL, Resting BP: {trestbps} mm Hg",
                        f"- Max HR: {thalach}, Oldpeak: {oldpeak}",
                        f"- Exercise-induced angina: {exang}, FBS>120: {fbs}, Chest pain: {cp}",
                        f"- Resting ECG: {restecg}, ST slope: {slope}",
                    ]
                    summary = (
                        "Heart disease risk result:\n" + risk_text +
                        "\n\n**Inputs used**\n" + "\n".join(explain) +
                        "\n\n⚠️ This is for education only. Not medical advice."
                    )
                    st.session_state.last_risk_context = summary

                except Exception as e:
                    st.error(f"Heart prediction failed: {e}")

            _render_risk_followup("heart_risk_chat", vector_store)

    # =========================
    #         DIABETES
    # =========================
    with tab_diab:
        if not have_diab:
            st.info("Diabetes model not loaded.")
        else:
            st.subheader("Diabetes Risk")

            c1, c2, c3 = st.columns(3)
            with c1:
                preg    = st.number_input("Pregnancies", min_value=0, max_value=20, value=2, step=1)
                glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120, step=1)
                bp      = st.number_input("Blood Pressure", min_value=0, max_value=220, value=72, step=1)
            with c2:
                skin    = st.number_input("Skin Thickness", min_value=0, max_value=100, value=23, step=1)
                insulin = st.number_input("Insulin", min_value=0, max_value=900, value=94, step=1)
                bmi     = st.number_input("BMI", min_value=10.0, max_value=70.0, value=28.0, step=0.1)
            with c3:
                dpf  = st.number_input("Diabetes Pedigree Fn", min_value=0.0, max_value=3.0, value=0.45, step=0.01)
                age2 = st.number_input("Age", min_value=0, max_value=120, value=35, step=1)

            X_diab = [[preg, glucose, bp, skin, insulin, bmi, dpf, age2]]

            if st.button("Predict Diabetes Risk", type="primary", key="predict_diab"):
                try:
                    model = models["diabetes"]
                    risk_pct, _ = predict_with_pipeline(model, X_diab)
                    risk_text = f"**Diabetes Risk:** {format_percent(risk_pct)}"
                    st.success(risk_text)

                    explain = [
                        f"- Glucose: {glucose}, BP: {bp}, BMI: {bmi}",
                        f"- Insulin: {insulin}, Skin thickness: {skin}",
                        f"- DPF: {dpf}, Pregnancies: {preg}, Age: {age2}",
                    ]
                    summary = (
                        "Diabetes risk result:\n" + risk_text +
                        "\n\n**Inputs used**\n" + "\n".join(explain) +
                        "\n\n⚠️ This is for education only. Not medical advice."
                    )
                    st.session_state.last_risk_context = summary

                except Exception as e:
                    st.error(f"Diabetes prediction failed: {e}")

            _render_risk_followup("diabetes_risk_chat", vector_store)




def render_browse_page(vector_store):
    """Render the browse topics page"""
    st.title("Browse Medical Topics")
    
    # Get available topics
    topics = get_medical_topics(vector_store)
    
    if topics:
        st.write(f"**{len(topics)} medical topics available**")
        
        # Search topics
        col1, col2 = st.columns([3, 1])
        with col1:
            topic_search = st.text_input("🔍 Search topics:", placeholder="Type to filter topics...")
        with col2:
            st.metric("Available Topics", len(topics))
        
        if topic_search:
            filtered_topics = [t for t in topics if topic_search.lower() in t.lower()]
            st.info(f"Found {len(filtered_topics)} topics matching '{topic_search}'")
        else:
            filtered_topics = topics
        
        # Display topics in columns with pagination
        topics_per_page = 15
        total_pages = (len(filtered_topics) + topics_per_page - 1) // topics_per_page
        
        if total_pages > 1:
            page = st.selectbox("Select page:", range(1, total_pages + 1)) - 1
        else:
            page = 0
        
        start_idx = page * topics_per_page
        end_idx = min(start_idx + topics_per_page, len(filtered_topics))
        current_topics = filtered_topics[start_idx:end_idx]
        
        # Display topics in a grid
        cols = st.columns(3)
        for i, topic in enumerate(current_topics):
            col_idx = i % 3
            with cols[col_idx]:
                # Card-like display with blue left-border on hover
                st.markdown(
                    f"<div class='topic-card'>{topic}</div>",
                    unsafe_allow_html=True,
                )

                # Button to view topic details (filled-blue primary style)
                if st.button(
                    "View Details",
                    key=f"topic_{i}_{page}",
                    use_container_width=True,
                    type="primary",
                ):
                    st.session_state.selected_disease = topic
                    st.session_state.current_page = 'disease_detail'
                    st.rerun()
        
        # Show pagination info
        if total_pages > 1:
            st.markdown(f"Page {page + 1} of {total_pages} | Showing {start_idx + 1}-{end_idx} of {len(filtered_topics)} topics")
    
    else:
        st.error("No topics found in the database.")

def render_disease_detail_page(vector_store):
    """Render detailed page for a specific disease"""
    disease = st.session_state.selected_disease
    
    if not disease:
        st.error("No disease selected")
        return
    
    # Header with navigation
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title(disease)
    with col2:
        if st.button("⬅️ Back to Browse", use_container_width=True, key="back_to_browse_detail"):
            st.session_state.current_page = 'browse'
            st.session_state.selected_disease = None
            st.rerun()
    
    st.markdown("---")
    
    # Search for information about this disease
    with st.spinner(f"Loading comprehensive information about {disease}..."):
        results = search_documents(vector_store, disease, max_results=15)
    
    if results:
        # Stats
        col1, col2, col3 = st.columns(3)
        sources = {}
        for result in results:
            source = result['source']
            sources[source] = sources.get(source, 0) + 1
        
        with col1:
            st.metric("Total Documents", len(results))
        with col2:
            st.metric("CDC Sources", sources.get('CDC', 0))
        with col3:
            st.metric("WHO Sources", sources.get('WHO', 0))
        
        # AI Summary Section
        st.subheader("Comprehensive Summary")
        
        api_key = os.getenv('OPENAI_API_KEY')
        use_ai = api_key and api_key != 'your-api-key-here'
        
        if use_ai:
            with st.spinner("Generating comprehensive AI summary..."):
                ai_summary = get_openai_summary(f"What is {disease}? Provide comprehensive information.", results)
            
            if ai_summary:
                st.markdown(ai_summary)
            else:
                st.warning("AI summary unavailable. Showing detailed keyword summary.")
                st.markdown(get_keyword_summary(disease, results))
        else:
            st.markdown(get_keyword_summary(disease, results))
        
        # Detailed Information Sections
        st.markdown("---")
        st.subheader("Detailed Information")
        
        # Group results by source
        cdc_results = [r for r in results if r['source'] == 'CDC']
        who_results = [r for r in results if r['source'] == 'WHO']
        
        # CDC Information
        if cdc_results:
            with st.expander(f"🏛️ CDC Information ({len(cdc_results)} documents)", expanded=True):
                for i, result in enumerate(cdc_results[:5], 1):
                    st.markdown(f"**Document {i}** (Relevance: {result['relevance_score']})")
                    st.markdown(result['text'])
                    st.markdown("---")
        
        # WHO Information
        if who_results:
            with st.expander(f"🌍 WHO Information ({len(who_results)} documents)", expanded=True):
                for i, result in enumerate(who_results[:5], 1):
                    st.markdown(f"**Document {i}** (Relevance: {result['relevance_score']})")
                    st.markdown(result['text'])
                    st.markdown("---")
        
        # Related Topics
        st.subheader("Related Topics")
        related_keywords = disease.split()[:3]  # Use first 3 words as related keywords
        
        for keyword in related_keywords:
            if len(keyword) > 3:
                if st.button(f"🔍 Search: {keyword}", key=f"related_{keyword}"):
                    st.session_state.search_query = keyword
                    st.session_state.current_page = 'search'
                    st.rerun()
    
    else:
        st.error(f"No information found for {disease}")
        st.info("This topic might not be available in our current database. Try searching with different keywords.")

def render_examples_page(vector_store):
    """Render the examples page"""
    st.title("Quick Examples")
    st.markdown("Click any example below to see how the search works:")
    
    examples = [
        ("Acanthamoeba Eye Infections", "acanthamoeba symptoms treatment", "Learn about this rare but serious eye infection caused by a microscopic organism"),
        ("Child Vaccination Safety", "vaccine children safety side effects", "Find information about vaccine safety, schedules, and side effects for children"),
        ("Tick-Borne Disease Prevention", "tick bite prevention lyme disease", "Discover how to prevent tick bites and tick-borne diseases like Lyme disease"),
        ("Food Poisoning Information", "food poisoning symptoms treatment prevention", "Learn about foodborne illnesses, symptoms, and prevention methods"),
        ("Refugee Health Screening", "refugee health screening requirements", "Information about health requirements and screening for refugees"),
        ("Harmful Algal Blooms", "harmful algal blooms health effects", "Learn about the health effects of harmful algal blooms in water"),
        ("Travel Health Advice", "travel health vaccination requirements", "Find travel health recommendations and vaccination requirements"),
        ("Workplace Safety Guidelines", "workplace health safety guidelines", "Occupational health and safety information for various industries")
    ]
    
    for i, (title, query, description) in enumerate(examples):
        with st.expander(f"🔍 {title}", expanded=False):
            st.markdown(f"**Description:** {description}")
            st.markdown(f"**Search Query:** `{query}`")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button(f"🚀 Try This Search", key=f"example_search_{i}"):
                    st.session_state.search_query = query
                    st.session_state.current_page = 'search'
                    st.rerun()
            
            with col2:
                if st.button(f"📋 Browse Related Topics", key=f"example_browse_{i}"):
                    st.session_state.current_page = 'browse'
                    st.rerun()

def render_history_page():
    """Render the search history page"""
    st.title("Search History")
    st.markdown(f"**Search history for user:** {st.session_state.username}")
    
    # Load user history
    user_history = load_user_history(st.session_state.username)
    
    if not user_history:
        st.info("🔍 No search history found. Start searching to build your history!")
        st.markdown("### 🚀 Quick Actions")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 Start Searching", use_container_width=True):
                st.session_state.current_page = 'search'
                st.rerun()
        with col2:
            if st.button("📤 Upload & Ask", use_container_width=True):
                st.session_state.current_page = 'upload'
                st.rerun()
        return
    
    # Filter options
    st.subheader("Filter History")
    col1, col2 = st.columns(2)
    
    with col1:
        search_type_filter = st.selectbox(
            "Filter by search type:",
            ["All", "ai_search", "document_analysis", "llamaindex_search", "keyword_search", "document_keyword_analysis"]
        )
    
    with col2:
        entries_to_show = st.selectbox("Show entries:", [10, 25, 50, 100], index=1)
    
    # Filter history
    filtered_history = user_history
    if search_type_filter != "All":
        filtered_history = [entry for entry in user_history if entry.get('search_type') == search_type_filter]
    
    # Sort by timestamp (newest first)
    filtered_history = sorted(filtered_history, key=lambda x: x.get('timestamp', ''), reverse=True)
    filtered_history = filtered_history[:entries_to_show]
    
    # Search within history
    search_term = st.text_input("🔍 Search within your history:", placeholder="Enter keywords to find specific searches...")
    if search_term:
        search_term_lower = search_term.lower()
        filtered_history = [
            entry for entry in filtered_history 
            if search_term_lower in entry.get('query', '').lower() or search_term_lower in entry.get('response', '').lower()
        ]
        st.info(f"Found {len(filtered_history)} entries matching '{search_term}'")
    
    # Display history
    st.subheader(f"Recent Searches ({len(filtered_history)} entries)")
    
    if not filtered_history:
        st.warning("No entries match your current filters.")
        return
    
    for i, entry in enumerate(filtered_history):
        timestamp = entry.get('timestamp', 'Unknown time')
        query = entry.get('query', 'No query')
        response = entry.get('response', 'No response')
        search_type = entry.get('search_type', 'unknown')
        full_length = entry.get('full_response_length', len(response))
        
        # Format timestamp
        try:
            dt = datetime.fromisoformat(timestamp)
            formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            formatted_time = timestamp
        
        # Search type emoji
        type_emoji = {
            'ai_search': '🤖',
            'document_analysis': '📄',
            'llamaindex_search': '🦙',
            'keyword_search': '🔍',
            'document_keyword_analysis': '📋'
        }
        
        with st.expander(f"{type_emoji.get(search_type, '🔍')} {query[:50]}{'...' if len(query) > 50 else ''} - {formatted_time}"):
            st.markdown(f"**🕒 Time:** {formatted_time}")
            st.markdown(f"**🔍 Search Type:** {search_type.replace('_', ' ').title()}")
            st.markdown(f"**❓ Query:** {query}")
            st.markdown(f"**💬 Response:** {response}")
            
            if full_length > len(response):
                st.info(f"📏 Full response was {full_length} characters (truncated for storage)")
            
            # Quick actions
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"🔄 Search Again", key=f"research_{i}"):
                    st.session_state.search_query = query
                    st.session_state.current_page = 'search'
                    st.rerun()
            with col2:
                if st.button(f"📋 Copy Query", key=f"copy_{i}"):
                    st.code(query, language="text")
                    st.success("Query copied to code block!")
    
    # Clear history option
    st.markdown("---")
    st.subheader("Manage History")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🗑️ Clear All History", type="secondary"):
            st.session_state.confirm_clear_history = True
    
    # Confirmation dialog
    if getattr(st.session_state, 'confirm_clear_history', False):
        st.warning("⚠️ Are you sure you want to clear all your search history? This action cannot be undone.")
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("✅ Yes, Clear", type="primary"):
                try:
                    history_file = os.path.join("history", "history.json")
                    if os.path.exists(history_file):
                        with open(history_file, 'r', encoding='utf-8') as f:
                            history_data = json.load(f)
                        
                        # Clear user's history
                        history_data[st.session_state.username] = []
                        
                        with open(history_file, 'w', encoding='utf-8') as f:
                            json.dump(history_data, f, indent=2, ensure_ascii=False)
                        
                        st.success("✅ Search history cleared successfully!")
                        st.session_state.confirm_clear_history = False
                        st.rerun()
                    else:
                        st.info("No history file found.")
                except Exception as e:
                    st.error(f"Failed to clear history: {e}")
        
        with col2:
            if st.button("❌ Cancel"):
                st.session_state.confirm_clear_history = False
                st.rerun()

def save_search_history(username: str, query: str, response: str, search_type: str = "search"):
    """Save user search history to JSON file"""
    try:
        # Create history directory if it doesn't exist
        history_dir = "history"
        if not os.path.exists(history_dir):
            os.makedirs(history_dir)
        
        # History file path
        history_file = os.path.join(history_dir, "history.json")
        
        # Load existing history or create new
        if os.path.exists(history_file):
            with open(history_file, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
        else:
            history_data = {}
        
        # Initialize user history if not exists
        if username not in history_data:
            history_data[username] = []
        
        # Create search entry
        search_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response[:500] + "..." if len(response) > 500 else response,  # Truncate long responses
            "search_type": search_type,  # 'search', 'upload', 'browse'
            "full_response_length": len(response)
        }
        
        # Add to user history (keep last 100 searches per user)
        history_data[username].append(search_entry)
        if len(history_data[username]) > 100:
            history_data[username] = history_data[username][-100:]
        
        # Save to file
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
            
        return True
        
    except Exception as e:
        st.error(f"Failed to save search history: {e}")
        return False

def load_user_history(username: str) -> List[Dict]:
    """Load search history for a specific user"""
    try:
        history_file = os.path.join("history", "history.json")
        
        if not os.path.exists(history_file):
            return []
        
        with open(history_file, 'r', encoding='utf-8') as f:
            history_data = json.load(f)
        
        return history_data.get(username, [])
        
    except Exception as e:
        st.error(f"Failed to load search history: {e}")
        return []

def main():
    """Main Streamlit app with navigation"""
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

        html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Page background */
        .stApp, [data-testid="stAppViewContainer"] {
            background-color: #F8FAFF;
        }

        /* Hide Streamlit default chrome (header bar, gradient decoration,
           toolbar, deploy button, footer, hamburger menu). */
        header[data-testid="stHeader"] { display: none !important; }
        [data-testid="stDecoration"] { display: none !important; }
        .st-emotion-cache-1dp5vir { display: none !important; }
        [data-testid="stToolbar"] { display: none !important; }
        [data-testid="stToolbarActions"] { display: none !important; }
        footer { display: none !important; }
        [data-testid="stDeployButton"] { display: none !important; }
        #MainMenu { visibility: hidden !important; }

        /* Hide Streamlit's "Running ..." status text in the top-right corner. */
        [data-testid="stStatusWidget"] { display: none !important; }
        .stSpinner > div { display: none !important; }

        /* Page header accent — thin blue stripe above page content */
        section.main > div.block-container,
        section[data-testid="stMain"] > div.block-container {
            border-top: 3px solid #1D4ED8;
            padding-top: 24px;
            background-color: transparent;
        }

        /* Page title — small blue dot accent (skipped on .no-title-dot) */
        section.main h1::before,
        section[data-testid="stMain"] h1::before {
            content: "";
            display: inline-block;
            width: 8px;
            height: 8px;
            background: #1D4ED8;
            border-radius: 50%;
            margin-right: 12px;
            vertical-align: middle;
        }
        section.main h1.no-title-dot::before,
        section[data-testid="stMain"] h1.no-title-dot::before {
            content: none !important;
        }

        /* Hide sidebar collapse/expand toggle button completely */
        [data-testid="collapsedControl"] { display: none !important; }
        button[kind="header"] { display: none !important; }
        .st-emotion-cache-1egp75f { display: none !important; }

        /* Remove the floating menu button if it was added */
        .floating-menu-btn { display: none !important; }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #FFFFFF;
        }
        [data-testid="stSidebar"] h1 {
            font-size: 20px !important;
            font-weight: 600 !important;
            color: #0F172A !important;
            margin-bottom: 12px !important;
        }
        [data-testid="stSidebar"] h1::before { content: none !important; }
        [data-testid="stSidebar"] hr {
            border: none;
            border-top: 1px solid #E2E8F0;
            margin: 12px 0;
        }
        [data-testid="stSidebar"] .stButton > button {
            text-align: left;
            justify-content: flex-start;
            font-family: 'Inter', sans-serif;
            font-size: 14px;
            padding: 10px 16px;
            background: transparent;
            border: none;
            color: #0F172A;
            width: 100%;
            font-weight: 400;
            border-left: 3px solid transparent;
            border-radius: 0 8px 8px 0;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            background-color: #F0F4FF;
            color: #1D4ED8;
        }
        [data-testid="stSidebar"] .stButton > button:focus,
        [data-testid="stSidebar"] .stButton > button:active {
            color: #1D4ED8;
            box-shadow: none;
        }
        /* Active sidebar nav button (marked via type="primary") */
        [data-testid="stSidebar"] .stButton > button[kind="primary"] {
            background-color: #F0F4FF !important;
            color: #1D4ED8 !important;
            border: none !important;
            border-left: 3px solid #1D4ED8 !important;
            border-radius: 0 8px 8px 0 !important;
            box-shadow: none !important;
            font-weight: 500 !important;
        }
        [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
            background-color: #E0E9FF !important;
            color: #1D4ED8 !important;
        }

        /* Global primary buttons (outside sidebar) */
        section.main .stButton > button[kind="primary"],
        section[data-testid="stMain"] .stButton > button[kind="primary"],
        button[kind="primaryFormSubmit"] {
            background-color: #1D4ED8 !important;
            color: #FFFFFF !important;
            border: 1px solid #1D4ED8 !important;
            border-radius: 8px !important;
        }
        section.main .stButton > button[kind="primary"]:hover,
        button[kind="primaryFormSubmit"]:hover {
            background-color: #1741B6 !important;
            border-color: #1741B6 !important;
        }

        /* Page titles */
        h1, .stApp h1 {
            font-size: 26px !important;
            font-weight: 600 !important;
            color: #0F172A !important;
            font-family: 'Inter', sans-serif !important;
        }

        /* Section headings */
        h2, h3, .stApp h2, .stApp h3 {
            font-size: 18px !important;
            font-weight: 500 !important;
            color: #0F172A !important;
            font-family: 'Inter', sans-serif !important;
        }

        /* Body text */
        section.main p, section.main li, .stMarkdown p, .stMarkdown li {
            font-size: 15px;
            color: #475569;
            line-height: 1.7;
        }

        /* Cards/containers */
        [data-testid="stExpander"] {
            border: 1px solid #E2E8F0 !important;
            border-radius: 12px !important;
        }

        /* Input fields */
        .stTextInput input,
        .stTextArea textarea,
        .stNumberInput input,
        .stSelectbox > div > div {
            border-radius: 8px !important;
            border: 1px solid #E2E8F0 !important;
        }
        .stTextInput input:focus,
        .stTextArea textarea:focus,
        .stNumberInput input:focus {
            border-color: #1D4ED8 !important;
            box-shadow: 0 0 0 3px rgba(29, 78, 216, 0.15) !important;
            outline: none !important;
        }

        /* Metric value emphasis (e.g., Available Topics count) */
        [data-testid="stMetricValue"] {
            color: #1D4ED8 !important;
            font-size: 32px !important;
            font-weight: 600 !important;
        }

        /* Welcome-state suggestion cards on the empty chat page.
           The .welcome-anchor div is placed immediately before the
           st.columns row, so :has() lets us scope these styles to
           just the three suggestion buttons. */
        [data-testid="element-container"]:has(.welcome-anchor)
            + [data-testid="stHorizontalBlock"]
            [data-testid="column"] .stButton > button,
        [data-testid="element-container"]:has(.welcome-anchor)
            + [data-testid="element-container"]
            [data-testid="column"] .stButton > button {
            background: #FFFFFF;
            color: #0F172A;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 16px;
            text-align: left;
            justify-content: flex-start;
            white-space: normal;
            height: auto;
            min-height: 88px;
            font-size: 14px;
            font-weight: 500;
            line-height: 1.5;
            transition: border-color 0.18s ease, box-shadow 0.18s ease,
                        transform 0.18s ease;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        [data-testid="element-container"]:has(.welcome-anchor)
            + [data-testid="stHorizontalBlock"]
            [data-testid="column"] .stButton > button:hover,
        [data-testid="element-container"]:has(.welcome-anchor)
            + [data-testid="element-container"]
            [data-testid="column"] .stButton > button:hover {
            border-color: #1D4ED8;
            color: #1D4ED8;
            box-shadow: 0 2px 8px rgba(29, 78, 216, 0.10);
        }

        /* Browse Topics cards */
        .topic-card {
            background-color: #F8FAFF;
            border: 1px solid #E2E8F0;
            border-left: 3px solid transparent;
            border-radius: 12px;
            padding: 16px 18px;
            margin: 5px 0;
            min-height: 88px;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            color: #0F172A;
            font-weight: 600;
            font-size: 14px;
            transition: border-color 0.18s ease, box-shadow 0.18s ease, transform 0.18s ease;
        }
        .topic-card:hover {
            border-left-color: #1D4ED8;
            box-shadow: 0 2px 8px rgba(29, 78, 216, 0.08);
        }
        </style>
    """, unsafe_allow_html=True)

    # One-time pre-login onboarding
    if 'did_onboard' not in st.session_state:
        st.session_state.did_onboard = False
    if not st.session_state.did_onboard:
        render_onboarding_prelogin()
        return
    
    # Check authentication first
    if not st.session_state.authenticated:
        render_auth_page()
        return
    
    # Centered loading splash for the (cold-start) medical search init.
    # load_medical_search / load_llamaindex_search are @st.cache_resource so
    # only the very first authenticated render pays the load cost; we gate
    # on a session flag so subsequent reruns don't flash a spinner.
    load_placeholder = st.empty()
    if not st.session_state.get('_loaders_ready', False):
        with load_placeholder.container():
            _, mid_col, _ = st.columns([1, 2, 1])
            with mid_col:
                st.markdown(
                    "<div style='text-align:center;padding:120px 0 16px 0;'>"
                    "<div style='font-size:24px;font-weight:600;color:#0F172A;"
                    "font-family:Inter,sans-serif;'>Loading MediAid AI...</div>"
                    "<div style='font-size:14px;color:#64748B;margin-top:8px;'>"
                    "Initializing medical search systems</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
                with st.spinner("Loading MediAid AI..."):
                    vector_store = load_medical_search()
                    llamaindex_query_engine = load_llamaindex_search()
        st.session_state['_loaders_ready'] = True
    else:
        vector_store = load_medical_search()
        llamaindex_query_engine = load_llamaindex_search()

    # Replace the splash with the real app once loading is complete.
    load_placeholder.empty()

    # Store in session state for access across pages
    if 'llamaindex_engine' not in st.session_state:
        st.session_state.llamaindex_engine = llamaindex_query_engine

    if not vector_store:
        st.error("Failed to load medical database. Please check the FAISS index file.")
        return
    
    # Render navigation (includes logout button at bottom)
    render_navigation()
    
    # Render current page
    if st.session_state.current_page == 'home':
        render_home_page(vector_store)
    elif st.session_state.current_page == 'search':
        render_search_page(vector_store)
    elif st.session_state.current_page == 'upload':
        render_upload_page(vector_store)
    elif st.session_state.current_page == 'risk':
        render_risk_page(vector_store)
    elif st.session_state.current_page == 'browse':
        render_browse_page(vector_store)
    elif st.session_state.current_page == 'disease_detail':
        render_disease_detail_page(vector_store)
    elif st.session_state.current_page == 'history':
        render_history_page()
    elif st.session_state.current_page == 'faqs':
        render_examples_page(vector_store)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**MediAid AI** | Medical information from CDC and WHO databases | "
        "Always consult healthcare professionals for medical advice | "
        "Medical queries only — Non-medical content is prohibited"
    )

if __name__ == "__main__":
    main()
