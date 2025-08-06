#!/usr/bin/env python3
"""
Medical Knowledge Search System - No OpenAI Required
Advanced keyword-based search through your medical knowledge base
"""

import os
import sys
import re
from typing import List, Dict, Tuple
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

class MedicalSearch:
    """Medical knowledge search without OpenAI dependency"""
    
    def __init__(self, faiss_index_path: str):
        """Initialize the medical search system"""
        
        from faiss_utils import FAISSVectorStore
        
        self.vector_store = FAISSVectorStore()
        self.vector_store.load_index(faiss_index_path)
        
        print(f"✅ Loaded {self.vector_store.index.ntotal} medical documents")
        
        # Build search index
        self._build_search_index()
    
    def _build_search_index(self):
        """Build keyword search index"""
        
        self.search_index = {}
        
        for i, text in enumerate(self.vector_store.texts):
            # Extract keywords and create searchable index
            words = re.findall(r'\b\w+\b', text.lower())
            
            for word in words:
                if len(word) > 2:  # Skip very short words
                    if word not in self.search_index:
                        self.search_index[word] = []
                    self.search_index[word].append(i)
        
        print(f"✅ Built search index with {len(self.search_index)} keywords")
    
    def search(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Search for medical information using keywords
        
        Args:
            query: Search query (keywords)
            max_results: Maximum number of results
            
        Returns:
            List of search results with relevance scores
        """
        
        # Parse query into keywords
        query_words = re.findall(r'\b\w+\b', query.lower())
        query_words = [w for w in query_words if len(w) > 2]
        
        if not query_words:
            return []
        
        # Find documents containing query words
        doc_scores = {}
        
        for word in query_words:
            if word in self.search_index:
                for doc_idx in self.search_index[word]:
                    if doc_idx not in doc_scores:
                        doc_scores[doc_idx] = 0
                    
                    # Score based on word frequency and importance
                    text = self.vector_store.texts[doc_idx].lower()
                    word_count = text.count(word)
                    
                    # Boost score for title/heading matches
                    if word in text[:100].lower():
                        word_count *= 2
                    
                    doc_scores[doc_idx] += word_count
        
        # Sort by relevance
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Format results
        results = []
        for doc_idx, score in sorted_docs[:max_results]:
            result = {
                'text': self.vector_store.texts[doc_idx],
                'metadata': self.vector_store.metadata[doc_idx],
                'relevance_score': score,
                'source': self.vector_store.metadata[doc_idx].get('source', 'Unknown')
            }
            results.append(result)
        
        return results
    
    def get_medical_topics(self) -> List[str]:
        """Get list of available medical topics"""
        
        topics = set()
        for meta in self.vector_store.metadata:
            text = meta.get('text', '')
            if 'Disease:' in text:
                topic = text.split('Disease:')[1].split('\n')[0].strip()
                if topic:
                    topics.add(topic)
        
        return sorted(list(topics))
    
    def search_by_topic(self, topic: str) -> List[Dict]:
        """Search for documents about a specific medical topic"""
        
        results = []
        for i, meta in enumerate(self.vector_store.metadata):
            text = meta.get('text', '')
            if topic.lower() in text.lower():
                result = {
                    'text': self.vector_store.texts[i],
                    'metadata': meta,
                    'source': meta.get('source', 'Unknown')
                }
                results.append(result)
        
        return results
    
    def get_summary(self, results: List[Dict]) -> str:
        """Generate a simple summary from search results"""
        
        if not results:
            return "No relevant information found."
        
        # Extract key information
        sources = set()
        key_points = []
        
        for result in results[:3]:  # Use top 3 results
            text = result['text']
            source = result['source']
            sources.add(source)
            
            # Extract first sentence or key information
            sentences = text.split('. ')
            if sentences:
                key_points.append(sentences[0][:200])
        
        # Build summary
        summary = f"Based on {len(results)} documents from {', '.join(sources)}:\n\n"
        
        for i, point in enumerate(key_points, 1):
            summary += f"{i}. {point.strip()}...\n"
        
        summary += f"\nRecommendation: Consult healthcare professionals for specific medical advice."
        
        return summary

def interactive_medical_search():
    """Interactive medical search interface"""
    
    print("🏥 MEDICAL KNOWLEDGE SEARCH SYSTEM")
    print("="*50)
    print("Search through 5,489 CDC and WHO medical documents")
    print("Type 'help' for commands, 'quit' to exit")
    print("="*50)
    
    try:
        # Initialize search system
        search_system = MedicalSearch("rag_data/medical_embeddings.index")
        
        print(f"\n💡 Available commands:")
        print(f"   search <query>     - Search by keywords")
        print(f"   topic <topic>      - Search by medical topic")
        print(f"   topics             - List available topics")
        print(f"   help               - Show this help")
        print(f"   quit               - Exit")
        
        while True:
            try:
                user_input = input("\n🔍 Enter command: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                elif user_input.lower() == 'help':
                    print(f"\n📋 Example searches:")
                    print(f"   search acanthamoeba symptoms")
                    print(f"   search vaccine children")
                    print(f"   topic Acanthamoeba Infections")
                    print(f"   topics")
                
                elif user_input.lower() == 'topics':
                    print(f"\n🦠 Available medical topics:")
                    topics = search_system.get_medical_topics()
                    for i, topic in enumerate(topics[:20], 1):
                        print(f"   {i:2d}. {topic}")
                    if len(topics) > 20:
                        print(f"   ... and {len(topics) - 20} more topics")
                
                elif user_input.lower().startswith('search '):
                    query = user_input[7:].strip()
                    if query:
                        print(f"\n🔍 Searching for: '{query}'")
                        results = search_system.search(query, max_results=5)
                        
                        if results:
                            print(f"\n📋 Found {len(results)} relevant documents:")
                            
                            for i, result in enumerate(results, 1):
                                print(f"\n   {i}. Source: {result['source']} (Score: {result['relevance_score']})")
                                text = result['text'][:200] + "..." if len(result['text']) > 200 else result['text']
                                print(f"      {text}")
                            
                            # Generate summary
                            summary = search_system.get_summary(results)
                            print(f"\n📝 Summary:")
                            print(summary)
                        else:
                            print("❌ No results found. Try different keywords.")
                
                elif user_input.lower().startswith('topic '):
                    topic = user_input[6:].strip()
                    if topic:
                        print(f"\n🦠 Searching for topic: '{topic}'")
                        results = search_system.search_by_topic(topic)
                        
                        if results:
                            print(f"\n📋 Found {len(results)} documents about '{topic}':")
                            
                            for i, result in enumerate(results[:3], 1):
                                print(f"\n   {i}. Source: {result['source']}")
                                text = result['text'][:250] + "..." if len(result['text']) > 250 else result['text']
                                print(f"      {text}")
                        else:
                            print(f"❌ No documents found for topic '{topic}'")
                
                else:
                    print("❌ Unknown command. Type 'help' for available commands.")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n👋 Thank you for using the Medical Knowledge Search System!")
        
    except Exception as e:
        print(f"❌ Failed to initialize search system: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    interactive_medical_search()
