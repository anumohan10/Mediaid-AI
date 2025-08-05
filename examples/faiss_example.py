#!/usr/bin/env python3
"""
Example script demonstrating how to use FAISS with embedded data.
This shows the complete workflow from loading embeddings to performing searches.
"""

import json
import numpy as np
import os
from pathlib import Path

# Install required packages first:
# pip install faiss-cpu numpy

def load_embeddings_simple_example():
    """Simple example of loading and using embeddings with FAISS."""
    
    print("=== Simple FAISS Example ===\n")
    
    # 1. Load embeddings from your JSON file
    embedding_file = "rag_data/embedded/cdc_embeddings.json"
    
    if not os.path.exists(embedding_file):
        print(f"Embedding file {embedding_file} not found.")
        print("Please make sure you have generated embeddings first.")
        return
    
    print(f"Loading embeddings from {embedding_file}...")
    
    with open(embedding_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract embeddings and texts
    embeddings = []
    texts = []
    
    for item in data:
        if 'embedding' in item and item['embedding']:  # Skip empty embeddings
            embeddings.append(item['embedding'])
            texts.append(item.get('text', ''))
    
    if not embeddings:
        print("No valid embeddings found in the file.")
        return
    
    print(f"Loaded {len(embeddings)} embeddings")
    
    # 2. Convert to numpy array (required for FAISS)
    embeddings_array = np.array(embeddings, dtype=np.float32)
    print(f"Embeddings shape: {embeddings_array.shape}")
    
    # 3. Create FAISS index
    import faiss
    
    # Create a simple flat index (exact search)
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
    
    # Add embeddings to the index
    index.add(embeddings_array)
    print(f"FAISS index created with {index.ntotal} vectors")
    
    # 4. Perform a search
    print("\n=== Search Example ===")
    
    # Use the first embedding as a query (just for demonstration)
    query_vector = embeddings_array[0:1]  # First embedding as query
    
    # Search for top 3 similar items
    k = 3
    scores, indices = index.search(query_vector, k)
    
    print(f"Top {k} similar documents:")
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        print(f"\nRank {i+1}:")
        print(f"  Score: {score:.4f}")
        print(f"  Text: {texts[idx][:200]}...")  # First 200 characters
    
    # 5. Save the index for later use
    print("\n=== Saving Index ===")
    
    output_dir = Path("rag_data")
    output_dir.mkdir(exist_ok=True)
    
    index_path = output_dir / "example_faiss.index"
    faiss.write_index(index, str(index_path))
    
    # Save the texts separately (FAISS only stores vectors)
    texts_path = output_dir / "example_texts.json"
    with open(texts_path, 'w', encoding='utf-8') as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)
    
    print(f"Index saved to: {index_path}")
    print(f"Texts saved to: {texts_path}")
    
    # 6. Load the index back (to demonstrate persistence)
    print("\n=== Loading Saved Index ===")
    
    loaded_index = faiss.read_index(str(index_path))
    print(f"Loaded index with {loaded_index.ntotal} vectors")
    
    # Load texts
    with open(texts_path, 'r', encoding='utf-8') as f:
        loaded_texts = json.load(f)
    
    # Test search with loaded index
    test_scores, test_indices = loaded_index.search(query_vector, 2)
    print(f"Test search with loaded index:")
    for score, idx in zip(test_scores[0], test_indices[0]):
        print(f"  Score: {score:.4f}, Text: {loaded_texts[idx][:100]}...")


def advanced_faiss_example():
    """Advanced example with multiple embedding files and different index types."""
    
    print("\n\n=== Advanced FAISS Example ===\n")
    
    # Files to process
    embedding_files = [
        "rag_data/embedded/cdc_embeddings.json",
        "rag_data/embedded/who_embeddings.json"
    ]
    
    all_embeddings = []
    all_texts = []
    all_sources = []
    
    # Load all embedding files
    for file_path in embedding_files:
        if os.path.exists(file_path):
            print(f"Loading {file_path}...")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data:
                if 'embedding' in item and item['embedding']:
                    all_embeddings.append(item['embedding'])
                    all_texts.append(item.get('text', ''))
                    all_sources.append(item.get('source', 'Unknown'))
        else:
            print(f"File not found: {file_path}")
    
    if not all_embeddings:
        print("No embeddings found. Please generate embeddings first.")
        return
    
    print(f"Total embeddings loaded: {len(all_embeddings)}")
    
    # Convert to numpy
    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    
    # Create different types of FAISS indices
    dimension = embeddings_array.shape[1]
    
    print(f"\nCreating different FAISS index types...")
    
    # 1. Flat index (exact search)
    flat_index = faiss.IndexFlatIP(dimension)
    flat_index.add(embeddings_array)
    print(f"Flat index: {flat_index.ntotal} vectors")
    
    # 2. IVF index (approximate search, faster for large datasets)
    if len(all_embeddings) > 100:
        nlist = min(100, len(all_embeddings) // 10)  # Number of clusters
        quantizer = faiss.IndexFlatIP(dimension)
        ivf_index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        ivf_index.train(embeddings_array)  # IVF requires training
        ivf_index.add(embeddings_array)
        print(f"IVF index: {ivf_index.ntotal} vectors, {nlist} clusters")
    else:
        ivf_index = None
        print("Dataset too small for IVF index")
    
    # 3. HNSW index (hierarchical navigable small world - very fast)
    hnsw_index = faiss.IndexHNSWFlat(dimension, 16)  # 16 connections per node
    hnsw_index.add(embeddings_array)
    print(f"HNSW index: {hnsw_index.ntotal} vectors")
    
    # Compare search performance
    print(f"\n=== Comparing Search Performance ===")
    
    # Use first embedding as query
    query = embeddings_array[0:1]
    k = 5
    
    # Search with flat index
    scores_flat, indices_flat = flat_index.search(query, k)
    
    # Search with HNSW index
    scores_hnsw, indices_hnsw = hnsw_index.search(query, k)
    
    print(f"Flat index results:")
    for i, (score, idx) in enumerate(zip(scores_flat[0], indices_flat[0])):
        print(f"  {i+1}. Score: {score:.4f}, Source: {all_sources[idx]}")
    
    print(f"\nHNSW index results:")
    for i, (score, idx) in enumerate(zip(scores_hnsw[0], indices_hnsw[0])):
        print(f"  {i+1}. Score: {score:.4f}, Source: {all_sources[idx]}")
    
    # Save the best index
    output_path = "rag_data/medical_combined.index"
    faiss.write_index(flat_index, output_path)
    print(f"\nSaved combined index to: {output_path}")


def practical_search_example():
    """Practical example of how to perform semantic search."""
    
    print("\n\n=== Practical Search Example ===\n")
    
    # This example shows how you would integrate FAISS with your RAG system
    # For demonstration, we'll create a simple search function
    
    def semantic_search(query_text, embeddings, texts, sources, top_k=5):
        """
        Perform semantic search using text query.
        Note: In practice, you'd use OpenAI API to get query embedding.
        """
        # For this example, we'll use the first embedding as a "query"
        # In real usage, you'd get embedding for query_text using OpenAI API
        
        query_embedding = embeddings[0:1]  # Placeholder
        
        # Create index
        import faiss
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        
        # Search
        scores, indices = index.search(query_embedding, top_k)
        
        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({
                'score': float(score),
                'text': texts[idx],
                'source': sources[idx]
            })
        
        return results
    
    # Load some data for demonstration
    embedding_file = "rag_data/embedded/cdc_embeddings.json"
    
    if os.path.exists(embedding_file):
        with open(embedding_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        embeddings = []
        texts = []
        sources = []
        
        for item in data[:10]:  # Just use first 10 for demo
            if 'embedding' in item and item['embedding']:
                embeddings.append(item['embedding'])
                texts.append(item.get('text', ''))
                sources.append(item.get('source', 'CDC'))
        
        if embeddings:
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Perform search
            results = semantic_search(
                query_text="vaccine safety information",
                embeddings=embeddings_array,
                texts=texts,
                sources=sources,
                top_k=3
            )
            
            print("Search results for 'vaccine safety information':")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Score: {result['score']:.4f}")
                print(f"   Source: {result['source']}")
                print(f"   Text: {result['text'][:150]}...")
        else:
            print("No embeddings found for demo")
    else:
        print(f"Demo file {embedding_file} not found")


if __name__ == "__main__":
    try:
        import faiss
        import numpy as np
        
        # Run examples
        load_embeddings_simple_example()
        advanced_faiss_example()
        practical_search_example()
        
        print("\n" + "="*60)
        print("FAISS Examples Complete!")
        print("="*60)
        print("\nNext steps:")
        print("1. Install requirements: pip install -r requirements.txt")
        print("2. Set up OpenAI API key: export OPENAI_API_KEY='your-key'")
        print("3. Run: python scripts/build_faiss_index.py")
        print("4. Use the RAG system: python utils/rag.py")
        
    except ImportError as e:
        print(f"Missing required packages: {e}")
        print("Please install: pip install faiss-cpu numpy")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
