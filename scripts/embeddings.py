import json
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Load local embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load chunks
with open("rag_data/who_chunks.json", "r") as f:
    who_chunks = json.load(f)

with open("rag_data/cdc_chunks.json", "r") as f:
    cdc_chunks = json.load(f)

# Embed chunks
def embed_chunks(chunks, source_name):
    print(f"Embedding {source_name}:")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return [
        {
            "text": chunk["text"],
            "source": source_name,
            "embedding": emb.tolist()
        }
        for chunk, emb in zip(chunks, embeddings)
    ]

# Process and save
who_embedded = embed_chunks(who_chunks, "WHO")
cdc_embedded = embed_chunks(cdc_chunks, "CDC")

# Output path
os.makedirs("rag_data/embedded", exist_ok=True)
with open("rag_data/embedded/who_embeddings.json", "w") as f:
    json.dump(who_embedded, f, indent=2)

with open("rag_data/embedded/cdc_embeddings.json", "w") as f:
    json.dump(cdc_embedded, f, indent=2)

print(" Embedding completed and saved.")
