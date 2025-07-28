# scripts/collect_urls.py

import requests
from bs4 import BeautifulSoup
import json
import os
from requests_html import HTMLSession

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def collect_who_urls():
    base_url = "https://www.who.int"
    listing_url = f"{base_url}/news-room/fact-sheets"
    res = requests.get(listing_url)
    soup = BeautifulSoup(res.text, "html.parser")

    links = soup.select("ul#alphabetical-nav-filter a")
    urls = [base_url + a['href'] for a in links if a.get('href', '').startswith("/news-room/fact-sheets")]
    urls = list(set(urls))

    with open(f"{DATA_DIR}/who_urls.json", "w") as f:
        json.dump(urls, f, indent=2)
    print(f"Saved {len(urls)} WHO URLs to who_urls.json")

def collect_cdc_urls():
    session = HTMLSession()
    url = "https://www.cdc.gov/health-topics.html#cdc-atozlist"
    response = session.get(url)
    response.html.render(timeout=30, sleep=2)

    urls = []
    for char in list("ABCDEFGHIJKLMNOPQRSTUVWXYZ#"):
        block = response.html.find(f'div.row.char-block[data-id="{char}"]', first=True)
        if not block:
            print(f"No block found for {char}")
            continue

        anchors = block.find("ul.unstyled-list.pl-0 li a")
        for a in anchors:
            href = a.attrs.get("href", "")
            if href.startswith("http"):
                urls.append(href)
            elif href.startswith("/"):
                urls.append("https://www.cdc.gov" + href)

    urls = list(set(urls))  # remove duplicates

    with open(f"{DATA_DIR}/cdc_urls.json", "w") as f:
        json.dump(urls, f, indent=2)

    print(f"Saved {len(urls)} CDC URLs from A–Z to cdc_urls.json")

if __name__ == "__main__":
    collect_who_urls()
    collect_cdc_urls()
