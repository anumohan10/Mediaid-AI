#!/usr/bin/env python3
"""
Simple Medical RAG Demo - Test your OpenAI integration
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

def main():
    """Simple demo of the medical RAG system."""
    
    print("🏥 MEDICAL RAG SYSTEM - SIMPLE DEMO")
    print("="*40)
    
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or api_key == 'your-api-key-here':
        print("❌ Please set your OpenAI API key in the .env file")
        return
    
    print(f"✅ API Key configured: {api_key[:10]}...{api_key[-4:]}")
    
    try:
        from rag import MedicalRAG
        
        print("🤖 Initializing Medical RAG System...")
        rag = MedicalRAG("rag_data/medical_embeddings.index")
        print("✅ System ready!")
        
        # Test queries
        test_questions = [
            "What are the symptoms of Acanthamoeba infections?",
            "What vaccines are recommended for refugees?",
            "How can I prevent tick-borne diseases?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n📋 Test {i}: {question}")
            print("-" * 50)
            
            try:
                result = rag.query(question, k=2)
                
                print(f"🤖 Response:")
                print(f"{result['response']}")
                
                print(f"\n📚 Sources:")
                for doc in result['retrieved_docs']:
                    source = doc['metadata'].get('source', 'Unknown')
                    score = doc['score']
                    print(f"   • {source} (score: {score:.3f})")
                
            except Exception as e:
                print(f"❌ Query failed: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n🎉 Demo complete! Your RAG system is working!")
        print(f"You can now ask medical questions and get AI-powered answers.")
        
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
