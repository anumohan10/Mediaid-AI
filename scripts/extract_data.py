import requests
from bs4 import BeautifulSoup
from requests_html import HTMLSession
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



def extract_cdc_content(url):
    try:
        session = HTMLSession()
        r = session.get(url)
        r.html.render(timeout=30, sleep=2)

        # Disease name from first <h1>
        h1_tag = r.html.find("h1", first=True)
        disease_name = h1_tag.text if h1_tag else "Unknown"

        h2_tags = r.html.find("h2")
        content_blocks = []

        for h2 in h2_tags:
            heading = h2.text.strip()
            next_elements = h2.element.xpath("following-sibling::*")
            content = []

            for el in next_elements:
                if el.tag == "h2":
                    break
                if el.tag == "p":
                    content.append(el.text_content().strip())

            if content:
                content_blocks.append({
                    "heading": heading,
                    "content": "\n".join(content)
                })

        return {
            "url": url,
            "disease_name": disease_name,
            "sections": content_blocks
        }

    except Exception as e:
        print(f"[CDC] Failed to extract {url}: {e}")
        return None


def main():
    # Load WHO URLs
    with open(f"{DATA_DIR}/who_urls.json") as f:
        who_urls = json.load(f)

    who_data = []
    for url in who_urls:
        print("Extracting WHO content from:", url)
        data = extract_who_content(url)
        if data:
            who_data.append(data)

    with open(f"{DATA_DIR}/who_data.json", "w") as f:
        json.dump(who_data, f, indent=2)
    print("Saved WHO content to who_data.json")

    # Load CDC URLs
    with open(f"{DATA_DIR}/cdc_urls.json") as f:
        cdc_urls = json.load(f)

    cdc_data = []
    for url in cdc_urls:
        print("Extracting CDC content from:", url)
        data = extract_cdc_content(url)
        if data:
            cdc_data.append(data)

    with open(f"{DATA_DIR}/cdc_data.json", "w") as f:
        json.dump(cdc_data, f, indent=2)
    print("Saved CDC content to cdc_data.json")

if __name__ == "__main__":
    main()
