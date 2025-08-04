#!/usr/bin/env python3
"""
Script to create FAISS index from CDC and WHO embeddings.
This script will load your existing embedding files and create a FAISS index for efficient similarity search.
"""

import os
import sys
from pathlib import Path

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from faiss_utils import create_faiss_index_from_embeddings, FAISSVectorStore

def main():
    """Main function to create FAISS index from embeddings."""
    
    # Define paths
    base_dir = Path(__file__).parent.parent
    embedding_dir = base_dir / "rag_data" / "embedded"
    output_dir = base_dir / "rag_data"
    
    # Embedding files
    embedding_files = [
        str(embedding_dir / "cdc_embeddings.json"),
        str(embedding_dir / "who_embeddings.json")
    ]
    
    # Check if files exist
    existing_files = []
    for file_path in embedding_files:
        if os.path.exists(file_path):
            existing_files.append(file_path)
            print(f"✓ Found: {file_path}")
        else:
            print(f"✗ Missing: {file_path}")
    
    if not existing_files:
        print("No embedding files found. Please ensure your embeddings are generated first.")
        return
    
    print(f"\nCreating FAISS index from {len(existing_files)} embedding files...")
    
    try:
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create FAISS index
        index_path = str(output_dir / "medical_embeddings.index")
        
        vector_store = create_faiss_index_from_embeddings(
            embedding_files=existing_files,
            output_path=index_path,
            index_type="flat"  # Use flat index for exact search
        )
        
        print(f"\n✓ FAISS index created successfully!")
        print(f"  - Index file: {index_path}")
        print(f"  - Metadata file: {index_path.replace('.index', '_metadata.json')}")
        print(f"  - Total vectors: {vector_store.index.ntotal}")
        
        # Test the index with a simple search
        print("\n--- Testing the index ---")
        if vector_store.texts:
            # Use the first embedding as a test query
            first_embedding = None
            with open(existing_files[0], 'r') as f:
                import json
                data = json.load(f)
                for item in data:
                    if 'embedding' in item and item['embedding']:
                        first_embedding = item['embedding']
                        break
            
            if first_embedding:
                results = vector_store.search(first_embedding, k=3)
                print(f"Test search returned {len(results)} results:")
                for result in results:
                    print(f"  Score: {result['score']:.4f}")
                    print(f"  Text: {result['text'][:100]}...")
                    print(f"  Source: {result['metadata'].get('source', 'Unknown')}")
                    print()
        
    except Exception as e:
        print(f"Error creating FAISS index: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
