import json
import os

def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def convert_to_chunks(data, source_name):
    chunks = []
    for record in data:
        disease = record.get("disease_name", "Unknown")
        url = record.get("url", "")
        for section in record.get("sections", []):
            heading = section.get("heading", "")
            content = section.get("content", "")
            chunk_text = f"Disease: {disease}\nSource: {source_name}\nHeading: {heading}\n{content}"
            chunks.append({"text": chunk_text})
    return chunks

def save_chunks(chunks, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

def main():
    # WHO
    who_data = load_data("cleaned/who_data_cleaned.json")
    who_chunks = convert_to_chunks(who_data, "WHO")
    save_chunks(who_chunks, "rag_data/who_chunks.json")

    # CDC
    cdc_data = load_data("cleaned/cdc_data_cleaned.json")
    cdc_chunks = convert_to_chunks(cdc_data, "CDC")
    save_chunks(cdc_chunks, "rag_data/cdc_chunks.json")

    print(f"✅ Saved {len(who_chunks)} WHO chunks and {len(cdc_chunks)} CDC chunks to 'rag_data/'")

if __name__ == "__main__":
    main()
