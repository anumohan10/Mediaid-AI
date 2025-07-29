import requests
from bs4 import BeautifulSoup
import json
import os

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


def extract_who_content(url):
    try:
        res = requests.get(url)
        soup = BeautifulSoup(res.text, "html.parser")

        # Get disease name
        disease_name_tag = soup.find("h1")
        disease_name = disease_name_tag.get_text(strip=True) if disease_name_tag else "Unknown"

        sections = []

        # 🔹 Extract Key Facts (outside <article>)
        key_facts_div = soup.find("div", class_="list-bold separator-line")
        if key_facts_div:
            li_tags = key_facts_div.find_all("li")
            key_facts = [li.get_text(strip=True) for li in li_tags if li.get_text(strip=True)]
            if key_facts:
                sections.append({
                    "heading": "Key Facts",
                    "content": "\n".join(key_facts)
                })

        # 🔹 Main body inside article
        article = soup.find("article", class_="sf-detail-body-wrapper")
        if not article:
            print(f"[WHO] Article not found on {url}")
            return None

        current_heading = None
        current_paras = []

        for tag in article.find_all(["h2", "p", "ul"], recursive=True):
            if tag.name == "h2":
                if current_heading and current_paras:
                    sections.append({
                        "heading": current_heading,
                        "content": "\n".join(current_paras)
                    })
                current_heading = tag.get_text(strip=True)
                current_paras = []
            else:
                text = tag.get_text(strip=True)
                if text:
                    current_paras.append(text)

        # Append last section
        if current_heading and current_paras:
            sections.append({
                "heading": current_heading,
                "content": "\n".join(current_paras)
            })

        return {
            "url": url,
            "disease_name": disease_name,
            "sections": sections
        }

    except Exception as e:
        print(f"[WHO] Failed to extract {url}: {e}")
        return None


def extract_cdc_content_bs4(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract disease name
        h1_tag = soup.find("h1")
        disease_name = h1_tag.get_text(strip=True) if h1_tag else "Unknown"

        # Extract h2 and following p tags
        sections = []
        current_heading = None
        content = []

        for tag in soup.find_all(["h2", "p"]):
            if tag.name == "h2":
                if current_heading and content:
                    sections.append({
                        "heading": current_heading,
                        "content": "\n".join(content)
                    })
                current_heading = tag.get_text(strip=True)
                content = []
            elif tag.name == "p" and current_heading:
                text = tag.get_text(strip=True)
                if text:
                    content.append(text)

        if current_heading and content:
            sections.append({
                "heading": current_heading,
                "content": "\n".join(content)
            })

        return {
            "url": url,
            "disease_name": disease_name,
            "sections": sections
        }

    except Exception as e:
        print(f"[CDC] Failed to extract {url}: {e}")
        return None


def main():
    # WHO Extraction
    who_path = os.path.join(DATA_DIR, "who_urls.json")
    with open(who_path) as f:
        who_urls = json.load(f)

    who_data = []
    for url in who_urls:
        print("Extracting WHO content from:", url)
        data = extract_who_content(url)
        if data:
            who_data.append(data)

    with open(os.path.join(DATA_DIR, "who_data.json"), "w") as f:
        json.dump(who_data, f, indent=2)
    print("Saved WHO content to who_data.json")

    # CDC Extraction (only first 2)
    cdc_path = os.path.join(DATA_DIR, "cdc_urls.json")
    with open(cdc_path) as f:
        cdc_urls = json.load(f)

    cdc_data = []
    for url in cdc_urls[:2]:  # Only process first 2
        print("Extracting CDC content from:", url)
        data = extract_cdc_content_bs4(url)
        if data:
            cdc_data.append(data)

    with open(os.path.join(DATA_DIR, "cdc_data.json"), "w") as f:
        json.dump(cdc_data, f, indent=2)
    print("Saved CDC content to cdc_data.json")


if __name__ == "__main__":
    main()
