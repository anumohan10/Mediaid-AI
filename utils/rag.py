import os
import json
import numpy as np
from typing import List, Dict, Optional
from openai import OpenAI
from faiss_utils import FAISSVectorStore
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class MedicalRAG:
    """
    Retrieval-Augmented Generation system for medical information using FAISS.
    """
    
    def __init__(self, faiss_index_path: str, openai_api_key: Optional[str] = None):
        """
        Initialize the Medical RAG system.
        
        Args:
            faiss_index_path: Path to the FAISS index file
            openai_api_key: OpenAI API key (optional, can be set via environment)
        """
        # Initialize OpenAI client
        api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not provided. Please set OPENAI_API_KEY in your .env file or pass it directly.")
        
        self.client = OpenAI(api_key=api_key)
        
        # Load configuration from environment variables
        self.embedding_model = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-ada-002')
        self.chat_model = os.getenv('OPENAI_CHAT_MODEL', 'gpt-3.5-turbo')
        self.temperature = float(os.getenv('OPENAI_TEMPERATURE', '0.1'))
        
        # Load FAISS vector store
        self.vector_store = FAISSVectorStore()
        if os.path.exists(faiss_index_path):
            self.vector_store.load_index(faiss_index_path)
            print(f"Loaded FAISS index with {self.vector_store.index.ntotal} vectors")
        else:
            raise FileNotFoundError(f"FAISS index not found at {faiss_index_path}")
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a text using OpenAI API.
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values
        """
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            raise
    
    def retrieve_relevant_docs(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents with metadata
        """
        # Get query embedding
        query_embedding = self.get_embedding(query)
        
        # Search in FAISS index
        results = self.vector_store.search(query_embedding, k=k)
        
        return results
    
    def generate_response(self, query: str, retrieved_docs: List[Dict]) -> str:
        """
        # Generate response using retrieved documents and OpenAI chat completion.
        
        Args:
            query: User query
            retrieved_docs: Documents retrieved from vector search
            
        Returns:
            Generated response
        """
        # Prepare context from retrieved documents
        context_parts = []
        for doc in retrieved_docs:
            source = doc['metadata'].get('source', 'Unknown')
            text = doc['text']
            context_parts.append(f"Source: {source}\n{text}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Create system prompt
        system_prompt = """You are a helpful medical AI assistant. Use the provided medical information to answer questions accurately and professionally. 

Important guidelines:
1. Base your answers on the provided medical information
2. If the information is insufficient, say so clearly
3. Always cite sources when available
4. Recommend consulting healthcare professionals for specific medical advice
5. Be precise and avoid speculation

Provided Medical Information:
{context}""".format(context=context)
        
        # Generate response
        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                max_tokens=1000,
                temperature=self.temperature
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating response: {e}")
            raise
    
    def query(self, question: str, k: int = 5) -> Dict:
        """
        Complete RAG pipeline: retrieve relevant docs and generate response.
        
        Args:
            question: User question
            k: Number of documents to retrieve
            
        Returns:
            Dictionary with response and retrieved documents
        """
        print(f"Processing query: {question}")
        
        # Retrieve relevant documents
        retrieved_docs = self.retrieve_relevant_docs(question, k=k)
        print(f"Retrieved {len(retrieved_docs)} relevant documents")
        
        # Generate response
        response = self.generate_response(question, retrieved_docs)
        
        return {
            'question': question,
            'response': response,
            'retrieved_docs': retrieved_docs,
            'num_docs_retrieved': len(retrieved_docs)
        }


# Example usage
if __name__ == "__main__":
    # Example of how to use the Medical RAG system
    
    # Initialize RAG system
    rag = MedicalRAG(
        faiss_index_path="rag_data/medical_embeddings.index",
        openai_api_key=os.getenv('OPENAI_API_KEY')
    )
    
    # Example queries
    test_queries = [
        "What are the symptoms of Acanthamoeba infections?",
        "How can I prevent tick-borne encephalitis?",
        "What vaccines are recommended for immigrants and refugees?",
        "What are the risks of harmful algal blooms?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print('='*50)
        
        result = rag.query(query)
        
        print(f"\nResponse:\n{result['response']}")
        print(f"\nBased on {result['num_docs_retrieved']} retrieved documents:")
        for i, doc in enumerate(result['retrieved_docs'][:3], 1):
            print(f"{i}. Score: {doc['score']:.4f} | Source: {doc['metadata'].get('source', 'Unknown')}")
            print(f"   Text preview: {doc['text'][:100]}...")