import json
import numpy as np
import faiss
import os
from typing import List, Dict, Tuple, Optional

class FAISSVectorStore:
    """
    A utility class for managing FAISS vector store operations.
    Handles loading embeddings, creating FAISS index, and performing similarity searches.
    """
    
    def __init__(self, embedding_dimension: int = 1536):
        """
        Initialize the FAISS vector store.
        
        Args:
            embedding_dimension: Dimension of the embeddings (default 1536 for OpenAI embeddings)
        """
        self.embedding_dimension = embedding_dimension
        self.index = None
        self.texts = []
        self.metadata = []
        
    def load_embeddings_from_json(self, json_file_path: str) -> Tuple[np.ndarray, List[str], List[Dict]]:
        """
        Load embeddings from JSON file.
        
        Args:
            json_file_path: Path to the JSON file containing embeddings
            
        Returns:
            Tuple of (embeddings_array, texts, metadata)
        """
        print(f"Loading embeddings from {json_file_path}")
        
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        embeddings = []
        texts = []
        metadata = []
        
        for item in data:
            if 'embedding' in item and item['embedding']:  # Check if embedding exists and is not empty
                embeddings.append(item['embedding'])
                texts.append(item.get('text', ''))
                
                # Store metadata (everything except embedding)
                meta = {k: v for k, v in item.items() if k != 'embedding'}
                metadata.append(meta)
        
        if not embeddings:
            raise ValueError(f"No valid embeddings found in {json_file_path}")
        
        embeddings_array = np.array(embeddings, dtype=np.float32)
        print(f"Loaded {len(embeddings)} embeddings with dimension {embeddings_array.shape[1]}")
        
        return embeddings_array, texts, metadata
    
    def create_index(self, embeddings: np.ndarray, index_type: str = "flat") -> None:
        """
        Create FAISS index from embeddings.
        
        Args:
            embeddings: NumPy array of embeddings
            index_type: Type of FAISS index ("flat", "ivf", "hnsw")
        """
        dimension = embeddings.shape[1]
        
        if index_type == "flat":
            # Simple flat index (exact search)
            self.index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine similarity)
        elif index_type == "ivf":
            # IVF index for faster approximate search
            nlist = min(100, embeddings.shape[0] // 10)  # Number of clusters
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            # Train the index
            self.index.train(embeddings)
        elif index_type == "hnsw":
            # HNSW index for very fast approximate search
            M = 16  # Number of connections
            self.index = faiss.IndexHNSWFlat(dimension, M)
            self.index.hnsw.efConstruction = 200
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        # Add embeddings to index
        self.index.add(embeddings)
        print(f"Created {index_type} FAISS index with {self.index.ntotal} vectors")
    
    def build_from_json_files(self, json_files: List[str], index_type: str = "flat") -> None:
        """
        Build FAISS index from multiple JSON files containing embeddings.
        
        Args:
            json_files: List of paths to JSON files
            index_type: Type of FAISS index to create
        """
        all_embeddings = []
        all_texts = []
        all_metadata = []
        
        for json_file in json_files:
            if os.path.exists(json_file):
                embeddings, texts, metadata = self.load_embeddings_from_json(json_file)
                all_embeddings.append(embeddings)
                all_texts.extend(texts)
                all_metadata.extend(metadata)
            else:
                print(f"Warning: File {json_file} not found")
        
        if not all_embeddings:
            raise ValueError("No valid embedding files found")
        
        # Combine all embeddings
        combined_embeddings = np.vstack(all_embeddings)
        
        # Store texts and metadata
        self.texts = all_texts
        self.metadata = all_metadata
        
        # Create index
        self.create_index(combined_embeddings, index_type)
    
    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict]:
        """
        Search for similar vectors in the FAISS index.
        
        Args:
            query_embedding: Query embedding as list of floats
            k: Number of top results to return
            
        Returns:
            List of dictionaries containing search results
        """
        if self.index is None:
            raise ValueError("Index not created. Call build_from_json_files first.")
        
        # Convert query to numpy array
        query_vector = np.array([query_embedding], dtype=np.float32)
        
        # Perform search
        scores, indices = self.index.search(query_vector, k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.texts):  # Valid index
                result = {
                    'rank': i + 1,
                    'score': float(score),
                    'text': self.texts[idx],
                    'metadata': self.metadata[idx]
                }
                results.append(result)
        
        return results
    
    def save_index(self, index_path: str) -> None:
        """
        Save the FAISS index to disk.
        
        Args:
            index_path: Path where to save the index
        """
        if self.index is None:
            raise ValueError("No index to save")
        
        faiss.write_index(self.index, index_path)
        
        # Save texts and metadata separately
        metadata_path = index_path.replace('.index', '_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                'texts': self.texts,
                'metadata': self.metadata
            }, f, ensure_ascii=False, indent=2)
        
        print(f"Index saved to {index_path}")
        print(f"Metadata saved to {metadata_path}")
    
    def load_index(self, index_path: str) -> None:
        """
        Load a FAISS index from disk.
        
        Args:
            index_path: Path to the saved index
        """
        self.index = faiss.read_index(index_path)
        
        # Load texts and metadata
        metadata_path = index_path.replace('.index', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.texts = data.get('texts', [])
                self.metadata = data.get('metadata', [])
        
        print(f"Index loaded from {index_path}")
        print(f"Loaded {self.index.ntotal} vectors")


def create_faiss_index_from_embeddings(embedding_files: List[str], 
                                     output_path: str, 
                                     index_type: str = "flat") -> FAISSVectorStore:
    """
    Convenience function to create FAISS index from embedding files.
    
    Args:
        embedding_files: List of paths to JSON files containing embeddings
        output_path: Path where to save the index
        index_type: Type of FAISS index ("flat", "ivf", "hnsw")
        
    Returns:
        FAISSVectorStore instance
    """
    # Create vector store
    vector_store = FAISSVectorStore()
    
    # Build index from JSON files
    vector_store.build_from_json_files(embedding_files, index_type)
    
    # Save index
    vector_store.save_index(output_path)
    
    return vector_store


# Example usage
if __name__ == "__main__":
    # Example of how to use the FAISS utilities
    
    # Define paths to your embedding files
    embedding_files = [
        "rag_data/embedded/cdc_embeddings.json",
        "rag_data/embedded/who_embeddings.json"
    ]
    
    # Create FAISS index
    vector_store = create_faiss_index_from_embeddings(
        embedding_files=embedding_files,
        output_path="rag_data/faiss_index.index",
        index_type="flat"
    )
    
    print("FAISS index created successfully!")
