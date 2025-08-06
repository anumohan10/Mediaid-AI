#!/usr/bin/env python3
"""
Medical RAG Demo - Complete Question-Answering System
Ask medical questions and get AI-powered answers based on CDC and WHO data.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from faiss_utils import FAISSVectorStore

class MedicalRAGDemo:
    """
    Enhanced Medical RAG system for demonstration.
    """
    
    def __init__(self, faiss_index_path: str):
        """Initialize the Medical RAG system."""
        
        # Check for OpenAI API key
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            print("⚠️  OpenAI API key not found!")
            print("Please set your API key with:")
            print("   Windows: set OPENAI_API_KEY=your-key-here")
            print("   Linux/Mac: export OPENAI_API_KEY=your-key-here")
            print("\nOr run: setup_openai.bat")
            self.openai_available = False
        else:
            print("✅ OpenAI API key found!")
            self.openai_available = True
            
            # Initialize OpenAI client
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.openai_api_key)
                print("✅ OpenAI client initialized successfully!")
            except ImportError:
                print("❌ OpenAI package not found. Installing...")
                import subprocess
                subprocess.run([sys.executable, "-m", "pip", "install", "openai"], check=True)
                from openai import OpenAI
                self.client = OpenAI(api_key=self.openai_api_key)
            except Exception as e:
                print(f"❌ Error initializing OpenAI: {e}")
                self.openai_available = False
        
        # Load FAISS vector store
        self.vector_store = FAISSVectorStore()
        if os.path.exists(faiss_index_path):
            self.vector_store.load_index(faiss_index_path)
            print(f"✅ Loaded FAISS index with {self.vector_store.index.ntotal} medical documents")
        else:
            raise FileNotFoundError(f"FAISS index not found at {faiss_index_path}")
    
    def get_embedding(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """Get embedding for text using OpenAI API."""
        
        if not self.openai_available:
            raise ValueError("OpenAI API not available")
        
        try:
            response = self.client.embeddings.create(
                input=text,
                model=model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            raise
    
    def retrieve_relevant_docs(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve relevant documents for a query."""
        
        if not self.openai_available:
            print("🔍 Using keyword-based search (OpenAI not available)")
            return self._keyword_search(query, k)
        
        print(f"🔍 Getting embedding for query: '{query}'")
        
        # Get query embedding
        query_embedding = self.get_embedding(query)
        
        # Search in FAISS index
        results = self.vector_store.search(query_embedding, k=k)
        
        print(f"📊 Found {len(results)} relevant documents")
        return results
    
    def _keyword_search(self, query: str, k: int = 5) -> List[Dict]:
        """Fallback keyword-based search when OpenAI is not available."""
        
        query_words = query.lower().split()
        results = []
        
        for i, text in enumerate(self.vector_store.texts):
            text_lower = text.lower()
            score = 0
            
            for word in query_words:
                score += text_lower.count(word)
            
            if score > 0:
                results.append({
                    'rank': len(results) + 1,
                    'score': float(score),
                    'text': text,
                    'metadata': self.vector_store.metadata[i]
                })
        
        # Sort by score and return top k
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Re-rank the results
        for i, result in enumerate(results[:k]):
            result['rank'] = i + 1
        
        return results[:k]
    
    def generate_response(self, query: str, retrieved_docs: List[Dict], model: str = "gpt-3.5-turbo") -> str:
        """Generate response using retrieved documents."""
        
        if not self.openai_available:
            return self._generate_simple_response(query, retrieved_docs)
        
        # Prepare context from retrieved documents
        context_parts = []
        for doc in retrieved_docs:
            source = doc['metadata'].get('source', 'Unknown')
            text = doc['text']
            context_parts.append(f"Source: {source}\n{text}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Create system prompt
        system_prompt = f"""You are a helpful medical AI assistant. Use the provided medical information to answer questions accurately and professionally.

Important guidelines:
1. Base your answers on the provided medical information
2. If the information is insufficient, say so clearly
3. Always cite sources when available (CDC, WHO)
4. Recommend consulting healthcare professionals for specific medical advice
5. Be precise and avoid speculation
6. Format your response clearly with bullet points when appropriate

Provided Medical Information:
{context}"""
        
        print(f"🤖 Generating response using {model}...")
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating response: {e}")
            return self._generate_simple_response(query, retrieved_docs)
    
    def _generate_simple_response(self, query: str, retrieved_docs: List[Dict]) -> str:
        """Generate a simple response without OpenAI."""
        
        response_parts = [
            f"Based on your query about: {query}\n",
            f"Here are the most relevant medical documents I found:\n"
        ]
        
        for i, doc in enumerate(retrieved_docs[:3], 1):
            source = doc['metadata'].get('source', 'Unknown')
            text = doc['text'][:300] + "..." if len(doc['text']) > 300 else doc['text']
            score = doc.get('score', 0)
            
            response_parts.append(f"{i}. Source: {source} (Relevance: {score:.2f})")
            response_parts.append(f"   {text}\n")
        
        response_parts.append("\n⚠️  Note: This is a basic keyword-based search result. For more intelligent responses, please set up OpenAI API access.")
        response_parts.append("\n🏥 Always consult with healthcare professionals for medical advice.")
        
        return "\n".join(response_parts)
    
    def query(self, question: str, k: int = 5, model: str = "gpt-3.5-turbo") -> Dict:
        """Complete RAG pipeline: retrieve relevant docs and generate response."""
        
        print(f"\n{'='*60}")
        print(f"🔬 MEDICAL QUERY: {question}")
        print('='*60)
        
        # Retrieve relevant documents
        retrieved_docs = self.retrieve_relevant_docs(question, k=k)
        
        # Generate response
        response = self.generate_response(question, retrieved_docs, model=model)
        
        return {
            'question': question,
            'response': response,
            'retrieved_docs': retrieved_docs,
            'num_docs_retrieved': len(retrieved_docs),
            'openai_used': self.openai_available
        }

def demo_medical_queries():
    """Demonstrate the Medical RAG system with sample queries."""
    
    print("🏥 MEDICAL RAG SYSTEM DEMO")
    print("="*60)
    
    # Initialize RAG system
    try:
        rag = MedicalRAGDemo("rag_data/medical_embeddings.index")
    except Exception as e:
        print(f"❌ Error initializing RAG system: {e}")
        return
    
    # Sample medical queries
    sample_queries = [
        "What are the symptoms of Acanthamoeba infections?",
        "How can contact lens wearers prevent eye infections?",
        "What vaccines are recommended for immigrants and refugees?",
        "What are the health risks of harmful algal blooms?",
        "How is anthrax prevented in workplace settings?",
        "What should I know about food safety and contamination?"
    ]
    
    print(f"\n🧪 Running {len(sample_queries)} sample queries...\n")
    
    for i, query in enumerate(sample_queries, 1):
        print(f"\n🔬 Query {i}/{len(sample_queries)}")
        
        try:
            result = rag.query(query, k=3)
            
            print(f"\n📋 RESPONSE:")
            print(result['response'])
            
            print(f"\n📊 SOURCES USED:")
            for j, doc in enumerate(result['retrieved_docs'], 1):
                source = doc['metadata'].get('source', 'Unknown')
                score = doc.get('score', 0)
                text_preview = doc['text'][:100] + "..." if len(doc['text']) > 100 else doc['text']
                print(f"   {j}. {source} (Score: {score:.3f})")
                print(f"      {text_preview}")
            
            print(f"\n{'─'*60}")
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            continue
    
    return rag

def interactive_mode(rag_system):
    """Interactive mode for asking custom questions."""
    
    print(f"\n🎯 INTERACTIVE MEDICAL Q&A MODE")
    print("="*60)
    print("Ask medical questions! Type 'quit' to exit.\n")
    
    while True:
        try:
            question = input("🔬 Your medical question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            result = rag_system.query(question)
            
            print(f"\n📋 ANSWER:")
            print(result['response'])
            
            print(f"\n📊 Based on {result['num_docs_retrieved']} documents")
            print("─"*40)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main function to run the Medical RAG demo."""
    
    print("🏥 MEDICAL RAG SYSTEM WITH FAISS + OPENAI")
    print("="*60)
    
    # Check if FAISS index exists
    index_path = "rag_data/medical_embeddings.index"
    if not os.path.exists(index_path):
        print(f"❌ FAISS index not found at {index_path}")
        print("Please run: python scripts/build_faiss_index.py")
        return
    
    # Run demo
    rag_system = demo_medical_queries()
    
    if rag_system:
        # Ask user if they want interactive mode
        while True:
            choice = input("\n🎯 Would you like to try interactive mode? (y/n): ").lower().strip()
            if choice in ['y', 'yes']:
                interactive_mode(rag_system)
                break
            elif choice in ['n', 'no']:
                break
            else:
                print("Please enter 'y' or 'n'")
    
    print(f"\n🎉 Medical RAG Demo Complete!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
