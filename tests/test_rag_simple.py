#!/usr/bin/env python3
"""
Simple test script to verify your RAG system setup.
"""

import os
import sys
from config.openai_config import validate_openai_setup, get_openai_api_key

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'config'))

def test_openai_config():
    """Test OpenAI configuration."""
    print("🔧 Testing OpenAI Configuration...")
    
    try:
        from config.openai_config import validate_openai_setup, get_openai_api_key
        
        if validate_openai_setup():
            api_key = get_openai_api_key()
            masked_key = api_key[:8] + "*" * (len(api_key) - 12) + api_key[-4:] if len(api_key) > 12 else "***"
            print(f"✅ API Key found: {masked_key}")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"❌ Error testing config: {e}")
        return False

def test_faiss_index():
    """Test FAISS index loading."""
    print("\n🗂️  Testing FAISS Index...")
    
    index_path = "rag_data/medical_embeddings.index"
    if os.path.exists(index_path):
        try:
            sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
            from utils.faiss_utils import FAISSVectorStore
            
            vector_store = FAISSVectorStore()
            vector_store.load_index(index_path)
            
            print(f"✅ FAISS index loaded: {vector_store.index.ntotal} vectors")
            print(f"✅ Metadata loaded: {len(vector_store.metadata)} entries")
            return True
            
        except Exception as e:
            print(f"❌ Error loading FAISS index: {e}")
            return False
    else:
        print(f"❌ FAISS index not found at {index_path}")
        return False

def test_rag_system():
    """Test the complete RAG system."""
    print("\n🤖 Testing RAG System...")
    
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
        from utils.rag import MedicalRAG
        
        # Initialize RAG system
        rag = MedicalRAG("rag_data/medical_embeddings.index")
        print("✅ RAG system initialized successfully!")
        
        # Test with a simple query
        test_query = "What are Acanthamoeba infections?"
        print(f"\n🔍 Testing query: '{test_query}'")
        
        result = rag.query(test_query, k=2)
        
        print(f"✅ Query processed successfully!")
        print(f"📊 Retrieved {result['num_docs_retrieved']} documents")
        print(f"🤖 Response length: {len(result['response'])} characters")
        
        # Show a preview of the response
        response_preview = result['response'][:200] + "..." if len(result['response']) > 200 else result['response']
        print(f"\n📋 Response preview:")
        print(f"{response_preview}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing RAG system: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_setup_instructions():
    """Show setup instructions if tests fail."""
    print(f"\n📋 SETUP INSTRUCTIONS")
    print("="*50)
    
    print("1. 🔑 Set your OpenAI API key:")
    print("   Windows PowerShell: $env:OPENAI_API_KEY='your-key-here'")
    print("   Or edit: config/openai_config.py")
    print()
    
    print("2. 🗂️  Ensure FAISS index exists:")
    print("   Run: python scripts/build_faiss_index.py")
    print()
    
    print("3. 📦 Install requirements:")
    print("   Run: pip install -r requirements.txt")
    print()
    
    print("4. 🚀 Test the system:")
    print("   Run: python medical_rag_demo.py")

def main():
    """Main test function."""
    print("🧪 RAG SYSTEM SETUP TEST")
    print("="*50)
    
    # Run tests
    config_ok = test_openai_config()
    faiss_ok = test_faiss_index()
    
    if config_ok and faiss_ok:
        rag_ok = test_rag_system()
        
        if rag_ok:
            print(f"\n🎉 ALL TESTS PASSED!")
            print("Your RAG system is ready to use!")
            print()
            print("🚀 Next steps:")
            print("   • python medical_rag_demo.py - Full demo")
            print("   • python app.py - Web service")
            print("   • python demo_search.py - Search examples")
        else:
            print(f"\n⚠️  RAG system test failed")
            show_setup_instructions()
    else:
        print(f"\n⚠️  Basic setup incomplete")
        show_setup_instructions()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
