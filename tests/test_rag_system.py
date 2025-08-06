#!/usr/bin/env python3
"""
Test script for the Medical RAG system.
This script tests both the FAISS search and OpenAI integration.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

def test_openai_connection():
    """Test if OpenAI API key is working."""
    
    print("🔑 Testing OpenAI API Connection...")
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not found in .env file!")
        print("Please edit your .env file and set: OPENAI_API_KEY=sk-your-key-here")
        return False
    
    if not api_key.startswith('sk-'):
        print("❌ Invalid API key format. Should start with 'sk-'")
        return False
    
    print(f"✅ API Key found: {api_key[:10]}...{api_key[-4:]}")
    
    # Test OpenAI connection
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        # Simple test API call
        response = client.embeddings.create(
            input="test connection",
            model="text-embedding-ada-002"
        )
        
        print("✅ OpenAI API connection successful!")
        print(f"   Embedding dimension: {len(response.data[0].embedding)}")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI API connection failed: {e}")
        return False

def test_faiss_index():
    """Test if FAISS index loads correctly."""
    
    print("\n📚 Testing FAISS Index...")
    
    index_path = "rag_data/medical_embeddings.index"
    if not os.path.exists(index_path):
        print(f"❌ FAISS index not found at {index_path}")
        print("Please run: python scripts/build_faiss_index.py")
        return False
    
    try:
        from faiss_utils import FAISSVectorStore
        
        vector_store = FAISSVectorStore()
        vector_store.load_index(index_path)
        
        print(f"✅ FAISS index loaded successfully!")
        print(f"   Total vectors: {vector_store.index.ntotal}")
        print(f"   Available texts: {len(vector_store.texts)}")
        print(f"   Metadata entries: {len(vector_store.metadata)}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAISS index loading failed: {e}")
        return False

def test_full_rag_system():
    """Test the complete RAG system."""
    
    print("\n🤖 Testing Full RAG System...")
    
    try:
        from rag import MedicalRAG
        
        # Initialize RAG system
        rag = MedicalRAG("rag_data/medical_embeddings.index")
        
        print("✅ RAG system initialized successfully!")
        
        # Test with a simple query
        test_query = "What are the symptoms of Acanthamoeba infections?"
        print(f"\n🔍 Testing query: '{test_query}'")
        
        result = rag.query(test_query, k=3)
        
        print(f"\n📋 Query Results:")
        print(f"   Question: {result['question']}")
        print(f"   Retrieved docs: {result['num_docs_retrieved']}")
        print(f"\n💬 Response:")
        print(f"   {result['response']}")
        
        print(f"\n📄 Source Documents:")
        for i, doc in enumerate(result['retrieved_docs'], 1):
            print(f"   {i}. Source: {doc['metadata'].get('source', 'Unknown')}")
            print(f"      Score: {doc['score']:.4f}")
            print(f"      Text: {doc['text'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_usage_guide():
    """Show how to use the RAG system."""
    
    print(f"\n" + "="*60)
    print("🎉 RAG SYSTEM SETUP COMPLETE!")
    print("="*60)
    
    print("""
🚀 Your Medical RAG System is Ready! Here's how to use it:

1. **Simple Python Script**:
   ```python
   from utils.rag import MedicalRAG
   
   # Initialize RAG system
   rag = MedicalRAG("rag_data/medical_embeddings.index")
   
   # Ask questions
   result = rag.query("What vaccines are recommended for children?")
   print(result['response'])
   ```

2. **Interactive Session**:
   ```python
   python medical_rag_demo.py
   ```

3. **FastAPI Web Service**:
   ```python
   python app.py
   # Then visit: http://localhost:8000/docs
   ```

4. **Available Medical Topics**:
   ✅ 5,489 medical documents from CDC and WHO
   ✅ 1,068+ different health topics
   ✅ Eye infections, vaccines, immigrant health, food safety, etc.

📝 Example Questions You Can Ask:
   • "What are the symptoms of Acanthamoeba infections?"
   • "What vaccines are recommended for immigrants?"
   • "How can I prevent foodborne illnesses?"
   • "What are the signs of tick-borne encephalitis?"
   • "What health screenings do refugees need?"

🛠️ Configuration:
   Your settings are in .env file:
   - OPENAI_API_KEY: Set to your OpenAI API key
   - OPENAI_CHAT_MODEL: gpt-3.5-turbo (default)
   - OPENAI_EMBEDDING_MODEL: text-embedding-ada-002 (default)
   - OPENAI_TEMPERATURE: 0.1 (default, more factual responses)
""")

def main():
    """Main test function."""
    
    print("🔬 MEDICAL RAG SYSTEM TEST SUITE")
    print("="*50)
    
    # Test each component
    openai_ok = test_openai_connection()
    faiss_ok = test_faiss_index()
    
    if openai_ok and faiss_ok:
        rag_ok = test_full_rag_system()
        
        if rag_ok:
            show_usage_guide()
        else:
            print("\n❌ RAG system test failed. Check your configuration.")
    else:
        print("\n❌ Prerequisites not met:")
        if not openai_ok:
            print("   - Fix OpenAI API key configuration")
        if not faiss_ok:
            print("   - Build FAISS index first")

if __name__ == "__main__":
    main()
