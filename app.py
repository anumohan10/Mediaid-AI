#!/usr/bin/env python3
"""
Medical Knowledge Search API - FastAPI Web Service
Search through 5,489 CDC and WHO medical documents
"""

import os
import sys
import re
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from faiss_utils import FAISSVectorStore

app = FastAPI(
    title="Medical Knowledge Search API",
    description="Search through 5,489 CDC and WHO medical documents",
    version="1.0.0"
)

# Medical Search System
class MedicalSearchAPI:
    def __init__(self, faiss_index_path: str):
        self.vector_store = FAISSVectorStore()
        self.vector_store.load_index(faiss_index_path)
        self._build_search_index()
    
    def _build_search_index(self):
        """Build keyword search index"""
        self.search_index = {}
        
        for i, text in enumerate(self.vector_store.texts):
            words = re.findall(r'\b\w+\b', text.lower())
            
            for word in words:
                if len(word) > 2:
                    if word not in self.search_index:
                        self.search_index[word] = []
                    self.search_index[word].append(i)
    
    def search(self, query: str, max_results: int = 10):
        """Search for medical information"""
        query_words = re.findall(r'\b\w+\b', query.lower())
        query_words = [w for w in query_words if len(w) > 2]
        
        if not query_words:
            return []
        
        doc_scores = {}
        
        for word in query_words:
            if word in self.search_index:
                for doc_idx in self.search_index[word]:
                    if doc_idx not in doc_scores:
                        doc_scores[doc_idx] = 0
                    
                    text = self.vector_store.texts[doc_idx].lower()
                    word_count = text.count(word)
                    
                    if word in text[:100].lower():
                        word_count *= 2
                    
                    doc_scores[doc_idx] += word_count
        
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_idx, score in sorted_docs[:max_results]:
            result = {
                'text': self.vector_store.texts[doc_idx],
                'metadata': self.vector_store.metadata[doc_idx],
                'relevance_score': score,
                'source': self.vector_store.metadata[doc_idx].get('source', 'Unknown'),
                'preview': self.vector_store.texts[doc_idx][:200] + "..." if len(self.vector_store.texts[doc_idx]) > 200 else self.vector_store.texts[doc_idx]
            }
            results.append(result)
        
        return results
    
    def get_topics(self):
        """Get available medical topics"""
        topics = set()
        for meta in self.vector_store.metadata:
            text = meta.get('text', '')
            if 'Disease:' in text:
                topic = text.split('Disease:')[1].split('\n')[0].strip()
                if topic:
                    topics.add(topic)
        return sorted(list(topics))

# Initialize search system
search_system = None

@app.on_event("startup")
async def startup_event():
    """Initialize the search system"""
    global search_system
    
    index_path = "rag_data/medical_embeddings.index"
    if not os.path.exists(index_path):
        raise HTTPException(status_code=500, detail="FAISS index not found")
    
    try:
        search_system = MedicalSearchAPI(index_path)
        print(f"✅ Medical Search API started!")
        print(f"📚 Loaded {search_system.vector_store.index.ntotal} medical documents")
        print(f"🔍 Built search index with {len(search_system.search_index)} keywords")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize: {e}")

# Response models
class SearchResult(BaseModel):
    text: str
    source: str
    relevance_score: float
    preview: str

class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: List[SearchResult]
    summary: str

class TopicsResponse(BaseModel):
    total_topics: int
    topics: List[str]

# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def home():
    """Beautiful web interface for medical search"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Medical Knowledge Search</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container { 
                max-width: 1200px; 
                margin: 0 auto; 
                background: white; 
                border-radius: 20px; 
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }
            .header { 
                background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
                color: white; 
                padding: 40px; 
                text-align: center; 
            }
            .header h1 { font-size: 2.5em; margin-bottom: 10px; }
            .header p { font-size: 1.2em; opacity: 0.9; }
            .stats { 
                background: #f8f9fa; 
                padding: 20px; 
                display: flex; 
                justify-content: center; 
                gap: 40px;
                flex-wrap: wrap;
            }
            .stat { text-align: center; }
            .stat-number { font-size: 2em; font-weight: bold; color: #3498db; }
            .stat-label { color: #666; font-size: 0.9em; }
            .search-section { padding: 40px; }
            .search-box { 
                display: flex; 
                gap: 10px; 
                margin-bottom: 30px;
                flex-wrap: wrap;
            }
            .search-input { 
                flex: 1; 
                min-width: 300px;
                padding: 15px 20px; 
                font-size: 16px; 
                border: 2px solid #e1e8ed; 
                border-radius: 50px;
                outline: none;
                transition: all 0.3s ease;
            }
            .search-input:focus { border-color: #3498db; box-shadow: 0 0 0 3px rgba(52,152,219,0.1); }
            .search-btn { 
                padding: 15px 30px; 
                background: #3498db; 
                color: white; 
                border: none; 
                border-radius: 50px; 
                cursor: pointer; 
                font-size: 16px;
                transition: all 0.3s ease;
            }
            .search-btn:hover { background: #2980b9; transform: translateY(-2px); }
            .topics-btn { 
                padding: 15px 30px; 
                background: #27ae60; 
                color: white; 
                border: none; 
                border-radius: 50px; 
                cursor: pointer; 
                font-size: 16px;
                transition: all 0.3s ease;
            }
            .topics-btn:hover { background: #229954; transform: translateY(-2px); }
            .examples { 
                margin-bottom: 30px; 
                text-align: center;
            }
            .examples h3 { color: #2c3e50; margin-bottom: 15px; }
            .example-tag { 
                display: inline-block; 
                margin: 5px; 
                padding: 8px 16px; 
                background: #e8f4fd; 
                border: 1px solid #3498db;
                border-radius: 20px; 
                cursor: pointer; 
                font-size: 14px;
                transition: all 0.3s ease;
            }
            .example-tag:hover { 
                background: #3498db; 
                color: white; 
                transform: translateY(-2px);
            }
            #results { margin-top: 30px; }
            .loading { 
                text-align: center; 
                padding: 40px; 
                font-size: 18px; 
                color: #666;
            }
            .result-header { 
                background: #f8f9fa; 
                padding: 20px; 
                border-radius: 10px; 
                margin-bottom: 20px;
            }
            .result-header h3 { color: #2c3e50; margin-bottom: 10px; }
            .summary { 
                background: linear-gradient(135deg, #e8f5e8 0%, #d5f4e6 100%); 
                padding: 20px; 
                border-radius: 10px; 
                margin: 20px 0; 
                border-left: 4px solid #27ae60;
            }
            .result { 
                background: white; 
                padding: 20px; 
                margin: 15px 0; 
                border-radius: 10px; 
                border: 1px solid #e1e8ed;
                transition: all 0.3s ease;
            }
            .result:hover { 
                box-shadow: 0 5px 15px rgba(0,0,0,0.1); 
                transform: translateY(-2px);
            }
            .source-badge { 
                display: inline-block;
                background: #3498db; 
                color: white; 
                padding: 4px 12px; 
                border-radius: 15px; 
                font-size: 12px; 
                font-weight: bold; 
                margin-bottom: 10px;
            }
            .score { 
                color: #7f8c8d; 
                font-size: 0.9em; 
                float: right;
            }
            .result-text { 
                line-height: 1.6; 
                color: #2c3e50;
            }
            .topics-grid { 
                display: grid; 
                grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); 
                gap: 15px; 
                margin-top: 20px;
            }
            .topic-item { 
                background: #f8f9fa; 
                padding: 15px; 
                border-radius: 10px; 
                cursor: pointer; 
                transition: all 0.3s ease;
                border: 1px solid #e1e8ed;
            }
            .topic-item:hover { 
                background: #3498db; 
                color: white; 
                transform: translateY(-2px);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏥 Medical Knowledge Search</h1>
                <p>AI-Powered Search Through Medical Documents</p>
            </div>
            
            <div class="stats">
                <div class="stat">
                    <div class="stat-number">5,489</div>
                    <div class="stat-label">Medical Documents</div>
                </div>
                <div class="stat">
                    <div class="stat-number">1,068+</div>
                    <div class="stat-label">Medical Topics</div>
                </div>
                <div class="stat">
                    <div class="stat-number">CDC + WHO</div>
                    <div class="stat-label">Trusted Sources</div>
                </div>
            </div>
            
            <div class="search-section">
                <div class="search-box">
                    <input type="text" class="search-input" id="searchInput" 
                           placeholder="Search medical information... (e.g., 'acanthamoeba symptoms')" 
                           onkeypress="handleEnter(event)">
                    <button class="search-btn" onclick="search()">🔍 Search</button>
                    <button class="topics-btn" onclick="getTopics()">📚 Browse Topics</button>
                </div>
                
                <div class="examples">
                    <h3>💡 Try These Example Searches:</h3>
                    <span class="example-tag" onclick="searchExample('acanthamoeba symptoms')">acanthamoeba symptoms</span>
                    <span class="example-tag" onclick="searchExample('vaccine children safety')">vaccine children safety</span>
                    <span class="example-tag" onclick="searchExample('tick bite prevention')">tick bite prevention</span>
                    <span class="example-tag" onclick="searchExample('food poisoning symptoms')">food poisoning symptoms</span>
                    <span class="example-tag" onclick="searchExample('refugee health screening')">refugee health screening</span>
                    <span class="example-tag" onclick="searchExample('harmful algal blooms')">harmful algal blooms</span>
                </div>
                
                <div id="results"></div>
            </div>
        </div>
        
        <script>
            function handleEnter(event) {
                if (event.key === 'Enter') {
                    search();
                }
            }
            
            function searchExample(query) {
                document.getElementById('searchInput').value = query;
                search();
            }
            
            async function search() {
                const query = document.getElementById('searchInput').value.trim();
                if (!query) return;
                
                document.getElementById('results').innerHTML = '<div class="loading">🔍 Searching medical documents...</div>';
                
                try {
                    const response = await fetch(`/search?query=${encodeURIComponent(query)}&limit=8`);
                    const data = await response.json();
                    
                    let html = `
                        <div class="result-header">
                            <h3>Search Results for "${data.query}"</h3>
                            <p><strong>${data.total_results} relevant documents found</strong></p>
                        </div>
                    `;
                    
                    if (data.summary) {
                        html += `<div class="summary"><strong>💡 Summary:</strong><br>${data.summary}</div>`;
                    }
                    
                    data.results.forEach((result, index) => {
                        const sourceColor = result.source === 'CDC' ? '#e74c3c' : '#27ae60';
                        html += `
                            <div class="result">
                                <span class="source-badge" style="background: ${sourceColor};">📄 ${result.source}</span>
                                <span class="score">Relevance: ${result.relevance_score}</span>
                                <div class="result-text">${result.preview}</div>
                            </div>
                        `;
                    });
                    
                    document.getElementById('results').innerHTML = html;
                } catch (error) {
                    document.getElementById('results').innerHTML = `<div class="result"><strong>❌ Error:</strong> ${error.message}</div>`;
                }
            }
            
            async function getTopics() {
                document.getElementById('results').innerHTML = '<div class="loading">📚 Loading medical topics...</div>';
                
                try {
                    const response = await fetch('/topics');
                    const data = await response.json();
                    
                    let html = `
                        <div class="result-header">
                            <h3>📚 Available Medical Topics</h3>
                            <p><strong>${data.total_topics} topics available</strong> - Click any topic to search</p>
                        </div>
                        <div class="topics-grid">
                    `;
                    
                    data.topics.slice(0, 50).forEach((topic, index) => {
                        html += `<div class="topic-item" onclick="searchExample('${topic}')">${index + 1}. ${topic}</div>`;
                    });
                    
                    if (data.topics.length > 50) {
                        html += `<div class="topic-item" style="background: #3498db; color: white;">... and ${data.topics.length - 50} more topics</div>`;
                    }
                    
                    html += '</div>';
                    document.getElementById('results').innerHTML = html;
                } catch (error) {
                    document.getElementById('results').innerHTML = `<div class="result"><strong>❌ Error:</strong> ${error.message}</div>`;
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/search", response_model=SearchResponse)
async def search_medical_info(
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, description="Maximum number of results")
):
    """Search medical information by keywords"""
    
    if search_system is None:
        raise HTTPException(status_code=503, detail="Search system not initialized")
    
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    results = search_system.search(query, max_results=limit)
    
    # Generate simple summary
    if results:
        sources = set(r['source'] for r in results)
        summary = f"Found information from {', '.join(sources)}. "
        summary += "Always consult healthcare professionals for specific medical advice."
    else:
        summary = "No relevant information found. Try different keywords."
    
    return SearchResponse(
        query=query,
        total_results=len(results),
        results=[SearchResult(**r) for r in results],
        summary=summary
    )

@app.get("/topics", response_model=TopicsResponse)
async def get_medical_topics():
    """Get list of available medical topics"""
    
    if search_system is None:
        raise HTTPException(status_code=503, detail="Search system not initialized")
    
    topics = search_system.get_topics()
    
    return TopicsResponse(
        total_topics=len(topics),
        topics=topics
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    
    if search_system is None:
        raise HTTPException(status_code=503, detail="Search system not initialized")
    
    return {
        "status": "healthy",
        "total_documents": search_system.vector_store.index.ntotal,
        "total_keywords": len(search_system.search_index),
        "system": "Medical Knowledge Search API"
    }

@app.get("/stats")
async def get_stats():
    """Get knowledge base statistics"""
    
    if search_system is None:
        raise HTTPException(status_code=503, detail="Search system not initialized")
    
    sources = {}
    for meta in search_system.vector_store.metadata:
        source = meta.get('source', 'Unknown')
        sources[source] = sources.get(source, 0) + 1
    
    return {
        "total_documents": len(search_system.vector_store.texts),
        "total_keywords": len(search_system.search_index),
        "sources": sources,
        "topics_count": len(search_system.get_topics())
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Medical Knowledge Search API...")
    print("🌐 Web Interface: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("🔍 Ready to search 5,489 medical documents!")
    uvicorn.run(app, host="0.0.0.0", port=8000)