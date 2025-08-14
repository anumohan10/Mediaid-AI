#!/usr/bin/env python3
"""
MediAid AI - Interactive Medical Search Interface
Streamlit web app for searching medical information from CDC and WHO databases
"""

import streamlit as st
import os
import sys
import re
from typing import List, Dict, Optional
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# Configure Streamlit page
st.set_page_config(
    page_title="MediAid AI - Medical Search",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

@st.cache_resource
def load_medical_search():
    """Load the medical search system"""
    try:
        from utils.faiss_utils import FAISSVectorStore
        
        vector_store = FAISSVectorStore()
        vector_store.load_index("rag_data/medical_embeddings.index")
        
        return vector_store
    except Exception as e:
        st.error(f"Failed to load medical database: {e}")
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

def search_documents(vector_store, query: str, max_results: int = 10) -> List[Dict]:
    """Search for medical documents"""
    if not vector_store:
        return []
    
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
            'index': doc_idx
        }
        results.append(result)
    
    return results

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
    
    summary += "\n\n⚠️ **Important:** Always consult healthcare professionals for specific medical advice."
    
    return summary

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

def render_navigation():
    """Render navigation sidebar"""
    st.sidebar.title("🏥 MediAid AI")
    
    # Navigation buttons
    if st.sidebar.button("🏠 Home", use_container_width=True, key="nav_home"):
        st.session_state.current_page = 'home'
        st.session_state.selected_disease = None
        st.rerun()
    
    if st.sidebar.button("🔍 Search", use_container_width=True, key="nav_search"):
        st.session_state.current_page = 'search'
        st.session_state.selected_disease = None
        st.rerun()
    
    if st.sidebar.button("� Upload & Ask", use_container_width=True, key="nav_upload"):
        st.session_state.current_page = 'upload'
        st.session_state.selected_disease = None
        st.rerun()
    
    if st.sidebar.button("�📋 Browse Topics", use_container_width=True, key="nav_browse"):
        st.session_state.current_page = 'browse'
        st.session_state.selected_disease = None
        st.rerun()
    
    if st.sidebar.button("💡 Examples", use_container_width=True, key="nav_examples"):
        st.session_state.current_page = 'examples'
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
    
    # Removed debug information for production
    
    if api_key and api_key != 'your-api-key-here' and len(api_key) > 20:
        st.sidebar.success("🤖 AI Summaries: Enabled")
        use_ai = st.sidebar.checkbox("Use AI-powered summaries", value=True, key="ai_summaries_diabetes")
        
        # Removed test button for production
    else:
        st.sidebar.warning("🤖 AI Summaries: Disabled (API key not configured)")
        use_ai = False

def render_home_page(vector_store):
    """Render the home page"""
    st.title("🏥 MediAid AI - Medical Information Search")
    st.markdown("**Your comprehensive medical information assistant powered by CDC and WHO databases**")
    
    if vector_store:
        st.success(f"✅ Loaded {vector_store.index.ntotal} medical documents")
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Documents", "5,489")
    with col2:
        st.metric("Medical Topics", "1,068+")
    with col3:
        st.metric("Data Sources", "CDC + WHO")
    with col4:
        st.metric("Search Types", "3 Modes")
    
    st.markdown("---")
    
    # Features overview
    st.subheader("🌟 Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔍 Smart Search
        - Keyword-based medical search
        - AI-powered summaries
        - Relevance scoring
        - Source attribution
        
        ### 📋 Browse Topics
        - 1,068+ medical conditions
        - Dedicated disease pages
        - Detailed information
        - CDC and WHO data
        """)
    
    with col2:
        st.markdown("""
        ### 💡 Quick Examples
        - Pre-built searches
        - Common medical questions
        - Instant results
        - Educational content
        
        ### 🎯 Key Benefits
        - Evidence-based information
        - User-friendly interface
        - Mobile responsive
        - Always up-to-date
        """)
    
    # Quick search
    st.markdown("---")
    st.subheader("🚀 Quick Search")
    
    quick_query = st.text_input(
        "Enter a medical question:",
        placeholder="e.g., acanthamoeba symptoms, vaccine safety, food poisoning",
        key="home_search"
    )
    
    if quick_query:
        st.session_state.search_query = quick_query
        st.session_state.current_page = 'search'
        st.rerun()

def render_search_page(vector_store):
    """Render the chat-based search page"""
    st.title("💬 Chat with MediAid AI")
    st.markdown("Ask me anything about medical conditions, symptoms, treatments, or prevention. I can help with follow-up questions too!")
    
    # Check OpenAI status for AI responses
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    use_ai = api_key and api_key != 'your-api-key-here' and len(api_key) > 20
    
    if use_ai:
        st.sidebar.success("🤖 AI Chat: Enabled")
    else:
        st.sidebar.warning("🤖 AI Chat: Disabled (using keyword responses)")
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("📝 Chat History")
        for i, chat in enumerate(st.session_state.chat_history):
            # User message
            with st.container():
                st.markdown(f"**🧑 You:** {chat['user']}")
            
            # Assistant response
            with st.container():
                st.markdown(f"**🤖 MediAid AI:** {chat['assistant']}")
            
            if i < len(st.session_state.chat_history) - 1:
                st.markdown("---")
    
    # Chat input area
    st.subheader("💬 Ask a Question")
    
    # Use chat_input for a more natural chat experience
    user_message = st.chat_input("Type your medical question here...")
    
    # Alternative fallback input if chat_input is not available
    if not user_message:
        with st.form("chat_form", clear_on_submit=True):
            user_message = st.text_area(
                "Your message:",
                placeholder="e.g., What are the symptoms of diabetes? How can I prevent heart disease?",
                help="Ask any medical question - I can provide follow-up answers too!"
            )
            submitted = st.form_submit_button("Send 💬")
            if not submitted:
                user_message = None
    
    if user_message:
        # Search for relevant documents
        with st.spinner("Searching medical database and generating response..."):
            results = search_documents(vector_store, user_message, max_results=8)
        
        if results:
            # Generate conversational response
            if use_ai:
                ai_response = get_conversational_response(user_message, st.session_state.chat_history, results)
                
                if ai_response:
                    # Add to chat history
                    st.session_state.chat_history.append({
                        'user': user_message,
                        'assistant': ai_response
                    })
                    
                    # Show latest response immediately
                    st.success("Response generated!")
                    with st.container():
                        st.markdown(f"**🧑 You:** {user_message}")
                        st.markdown(f"**🤖 MediAid AI:** {ai_response}")
                    
                    # Add sources section
                    with st.expander("📚 Sources Used"):
                        for i, result in enumerate(results[:3], 1):
                            st.markdown(f"**{i}. {result['source']}** (Relevance: {result['relevance_score']})")
                            st.markdown(result['text'][:300] + "...")
                            if i < 3:
                                st.markdown("---")
                    
                    st.rerun()
                else:
                    st.error("AI response unavailable. Please try again.")
            else:
                # Fallback to keyword summary
                summary = get_keyword_summary(user_message, results)
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'user': user_message,
                    'assistant': summary
                })
                
                st.success("Response generated!")
                with st.container():
                    st.markdown(f"**🧑 You:** {user_message}")
                    st.markdown(f"**🤖 MediAid AI:** {summary}")
                
                st.rerun()
        else:
            st.warning("I couldn't find relevant information for your question. Please try rephrasing or ask about common medical topics.")
    
    # Clear chat button
    if st.session_state.chat_history:
        st.markdown("---")
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🗑️ Clear Chat", key="clear_chat_btn"):
                st.session_state.chat_history = []
                st.session_state.chat_context = []
                st.rerun()
        with col2:
            st.markdown(f"*{len(st.session_state.chat_history)} messages in chat*")

def render_upload_page(vector_store):
    """Render the Upload & Ask page for medical document analysis"""
    st.title("📄 Upload & Ask")
    st.markdown("Upload medical reports, prescriptions, or lab results and ask questions about them!")
    
    # Import OCR utilities
    try:
        from utils.ocr_utils import create_ocr_interface
        ocr_available = True
    except ImportError:
        ocr_available = False
    
    # Check OpenAI status for AI responses
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    use_ai = api_key and api_key != 'your-api-key-here' and len(api_key) > 20
    
    if use_ai:
        st.sidebar.success("🤖 AI Analysis: Enabled")
    else:
        st.sidebar.warning("🤖 AI Analysis: Disabled (using keyword responses)")
    
    # OCR Status indicator
    if ocr_available:
        st.sidebar.success("📄 Document Reading: Enabled")
    else:
        st.sidebar.error("📄 Document Reading: Disabled")
        st.error("❌ Document upload feature requires additional packages.")
        st.info("To enable document reading, install: `pip install easyocr opencv-python Pillow`")
        st.code("pip install easyocr opencv-python Pillow", language="bash")
        return
    
    # Document upload section
    st.subheader("📤 Upload Medical Document")
    
    # Create OCR interface
    extracted_text = create_ocr_interface()
    
    if extracted_text:
        # Store extracted text in session state
        st.session_state.document_context = extracted_text
        st.session_state.upload_page_document = True
        
        st.success("✅ Document processed successfully!")
        
        # Document analysis section
        if 'document_analysis' in st.session_state and st.session_state.document_analysis:
            st.subheader("🔍 Quick Analysis")
            analysis = st.session_state.document_analysis
            
            cols = st.columns(2)
            with cols[0]:
                if 'medications' in analysis:
                    st.markdown("**💊 Medications Found:**")
                    for med in analysis['medications'][:3]:
                        st.write(f"• {med}")
                
                if 'diagnoses' in analysis:
                    st.markdown("**🩺 Diagnoses Found:**")
                    for diag in analysis['diagnoses'][:3]:
                        st.write(f"• {diag}")
            
            with cols[1]:
                if 'lab_values' in analysis:
                    st.markdown("**📊 Lab Values Found:**")
                    for lab in analysis['lab_values'][:3]:
                        st.write(f"• {lab}")
                
                if 'vitals' in analysis:
                    st.markdown("**❤️ Vitals Found:**")
                    for vital in analysis['vitals'][:3]:
                        st.write(f"• {vital}")
        
        # Suggested questions
        st.subheader("💡 Suggested Questions")
        st.markdown("Click any question below to ask about your document:")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋 What medications are mentioned?", key="upload_med_question"):
                st.session_state.suggested_question = "What medications are mentioned in this document?"
            if st.button("🩺 What are the diagnoses?", key="upload_diag_question"):
                st.session_state.suggested_question = "What diagnoses or medical conditions are mentioned in this document?"
            if st.button("📊 What are the lab values?", key="upload_lab_question"):
                st.session_state.suggested_question = "What are the lab values or test results mentioned in this document?"
        
        with col2:
            if st.button("⚕️ Explain this report in simple terms", key="upload_explain_question"):
                st.session_state.suggested_question = "Can you explain what this medical report means in simple terms?"
            if st.button("⚠️ Are there any concerning values?", key="upload_concern_question"):
                st.session_state.suggested_question = "Are there any concerning values or findings in this document that I should be aware of?"
            if st.button("💡 What should I ask my doctor?", key="upload_doctor_question"):
                st.session_state.suggested_question = "Based on this document, what questions should I ask my doctor during my next visit?"
    
    # Chat section for document-specific questions
    if 'document_context' in st.session_state and st.session_state.document_context and 'upload_page_document' in st.session_state:
        st.markdown("---")
        st.subheader("💬 Ask Questions About Your Document")
        
        # Display uploaded document info
        with st.expander("📄 View Uploaded Document Text"):
            st.text_area("Extracted Text:", st.session_state.document_context, height=200, disabled=True)
        
        # Clear document button
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🗑️ Clear Document", key="clear_upload_doc"):
                if 'document_context' in st.session_state:
                    del st.session_state.document_context
                if 'document_analysis' in st.session_state:
                    del st.session_state.document_analysis
                if 'upload_page_document' in st.session_state:
                    del st.session_state.upload_page_document
                if 'upload_chat_history' in st.session_state:
                    del st.session_state.upload_chat_history
                st.success("Document cleared!")
                st.rerun()
        
        # Initialize upload-specific chat history
        if 'upload_chat_history' not in st.session_state:
            st.session_state.upload_chat_history = []
        
        # Display chat history for this document
        if st.session_state.upload_chat_history:
            st.markdown("**📝 Questions & Answers:**")
            for i, chat in enumerate(st.session_state.upload_chat_history):
                with st.container():
                    st.markdown(f"**🧑 You:** {chat['user']}")
                    st.markdown(f"**🤖 MediAid AI:** {chat['assistant']}")
                if i < len(st.session_state.upload_chat_history) - 1:
                    st.markdown("---")
        
        # Chat input
        user_message = None
        
        # Check for suggested questions
        if 'suggested_question' in st.session_state:
            user_message = st.session_state.suggested_question
            del st.session_state.suggested_question
        
        # Chat input
        if not user_message:
            user_message = st.chat_input("Ask a question about your uploaded document...")
        
        # Alternative input
        if not user_message:
            with st.form("upload_chat_form", clear_on_submit=True):
                user_message = st.text_area(
                    "Your question about the document:",
                    placeholder="e.g., What does this lab result mean? Are my medication dosages normal?",
                    help="Ask specific questions about your uploaded medical document"
                )
                submitted = st.form_submit_button("Ask 💬")
                if not submitted:
                    user_message = None
        
        if user_message:
            # Add debug information
            st.info(f"🔍 Processing your question: {user_message}")
            
            # Prepare enhanced query with document context
            document_context = st.session_state.document_context
            enhanced_query = f"""
            User uploaded a medical document with the following content:
            
            DOCUMENT CONTENT:
            {document_context}
            
            USER QUESTION ABOUT THE DOCUMENT:
            {user_message}
            
            Please answer the user's question based on the uploaded document content and provide additional relevant medical information if helpful.
            """
            
            # Search for relevant documents
            with st.spinner("Analyzing your document and searching medical database..."):
                try:
                    results = search_documents(vector_store, enhanced_query, max_results=5)
                    st.success(f"✅ Found {len(results)} relevant medical references")
                except Exception as e:
                    st.error(f"❌ Search error: {e}")
                    results = []
            
            if results:
                # Generate response
                if use_ai:
                    st.info("🤖 Generating AI response...")
                    try:
                        ai_response = get_conversational_response(enhanced_query, st.session_state.upload_chat_history, results, document_context)
                        
                        if ai_response:
                            # Add to upload-specific chat history
                            st.session_state.upload_chat_history.append({
                                'user': user_message,
                                'assistant': ai_response
                            })
                            
                            # Show response immediately
                            st.success("✅ Analysis complete!")
                            with st.container():
                                st.markdown(f"**🧑 You:** {user_message}")
                                st.info("📄 Response based on your uploaded document and medical database")
                                st.markdown(f"**🤖 MediAid AI:** {ai_response}")
                            
                            # Add sources
                            with st.expander("📚 Sources Used"):
                                st.markdown("**📄 Your Uploaded Document**")
                                st.markdown(document_context[:500] + "..." if len(document_context) > 500 else document_context)
                                st.markdown("---")
                                
                                for i, result in enumerate(results[:2], 1):
                                    st.markdown(f"**{i}. {result['source']}** (Relevance: {result['relevance_score']})")
                                    st.markdown(result['text'][:300] + "...")
                                    if i < 2:
                                        st.markdown("---")
                        else:
                            st.error("❌ AI analysis failed. Please try again.")
                            st.error("Debug: AI response was None or empty")
                    except Exception as e:
                        st.error(f"❌ AI analysis error: {e}")
                        import traceback
                        st.error(f"Debug: {traceback.format_exc()}")
                else:
                    # Fallback to keyword summary
                    st.info("🔍 Using keyword-based analysis...")
                    try:
                        summary = get_keyword_summary(user_message, results)
                        summary = f"Based on your uploaded document and medical database:\n\n{summary}"
                        
                        st.session_state.upload_chat_history.append({
                            'user': user_message,
                            'assistant': summary
                        })
                        
                        st.success("✅ Analysis complete!")
                        with st.container():
                            st.markdown(f"**🧑 You:** {user_message}")
                            st.markdown(f"**🤖 MediAid AI:** {summary}")
                    except Exception as e:
                        st.error(f"❌ Keyword analysis error: {e}")
            else:
                st.warning("No relevant medical information found. Please try rephrasing your question.")
                st.info("Debug: Search returned no results")
    
    else:
        # No document uploaded yet
        st.info("👆 Upload a medical document above to start asking questions about it!")
        
        # Example section
        st.subheader("📋 What You Can Upload")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **📄 Medical Reports:**
            • Blood test results
            • X-ray reports
            • MRI/CT scan reports
            • Pathology reports
            """)
            
            st.markdown("""
            **💊 Prescriptions:**
            • Medication lists
            • Prescription slips
            • Pharmacy labels
            """)
        
        with col2:
            st.markdown("""
            **🩺 Clinical Documents:**
            • Doctor's notes
            • Discharge summaries
            • Consultation letters
            • Medical certificates
            """)
            
            st.markdown("""
            **📊 Lab Results:**
            • Complete blood count
            • Lipid panels
            • Glucose tests
            • Hormone levels
            """)

def render_browse_page(vector_store):
    """Render the browse topics page"""
    st.title("📋 Browse Medical Topics")
    
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
                # Create a card-like display
                st.markdown(f"""
                <div style="
                    border: 1px solid #ddd; 
                    border-radius: 8px; 
                    padding: 10px; 
                    margin: 5px 0; 
                    background-color: #f9f9f9;
                    height: 100px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    text-align: center;
                ">
                    <strong>{topic}</strong>
                </div>
                """, unsafe_allow_html=True)
                
                # Button to view topic details
                if st.button(f"📖 View Details", key=f"topic_{i}_{page}", use_container_width=True):
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
        st.title(f"🦠 {disease}")
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
        st.subheader("📋 Comprehensive Summary")
        
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
        st.subheader("📚 Detailed Information")
        
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
        st.subheader("🔗 Related Topics")
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
    st.title("💡 Quick Examples")
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

def main():
    """Main Streamlit app with navigation"""
    
    # Load medical search system
    vector_store = load_medical_search()
    
    if not vector_store:
        st.error("Failed to load medical database. Please check the FAISS index file.")
        return
    
    # Render navigation
    render_navigation()
    
    # Render current page
    if st.session_state.current_page == 'home':
        render_home_page(vector_store)
    elif st.session_state.current_page == 'search':
        render_search_page(vector_store)
    elif st.session_state.current_page == 'upload':
        render_upload_page(vector_store)
    elif st.session_state.current_page == 'browse':
        render_browse_page(vector_store)
    elif st.session_state.current_page == 'disease_detail':
        render_disease_detail_page(vector_store)
    elif st.session_state.current_page == 'examples':
        render_examples_page(vector_store)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**MediAid AI** | Medical information from CDC and WHO databases | "
        "⚠️ Always consult healthcare professionals for medical advice"
    )

if __name__ == "__main__":
    main()
