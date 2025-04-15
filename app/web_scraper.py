# app/web_scraper.py
from keybert import KeyBERT
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

kw_model = KeyBERT()

def extract_keywords(text, top_n=5):
    keywords = kw_model.extract_keywords(
    text,
    keyphrase_ngram_range=(1, 2),
    stop_words='english',
    top_n=7,
    use_maxsum=True,
    nr_candidates=20
)

    return [kw[0] for kw in keywords]

import urllib.parse

def fetch_search_results(query):
    headers = {"User-Agent": "Mozilla/5.0"}
    # Encode the query for a Google search and use your custom prefix format
    encoded_query = urllib.parse.quote_plus(f"wikipedia%{query}")
    search_url = f"https://www.google.com/search?q={encoded_query}"
    
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    links = []

    for g in soup.find_all("div", class_="tF2Cxc"):
        link = g.find("a")
        if link and "wikipedia.org" in link["href"]:
            links.append(link["href"])

    return links[:5]  # Only top 5 Wikipedia links


def scrape_page_content(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    content = ""
    for p in soup.find_all("p"):
        content += p.get_text()
    return content


def compare_with_web(input_text):
    keywords = extract_keywords(input_text)
    matched_urls = []
    combined_scraped_text = ""

    print("🧠 Extracted keywords for search:")
    print("No of keywords detcted: ", len(keywords))
    for keyword in keywords:
        print("  🔍", keyword)
        urls = fetch_search_results(keyword)
        for url in urls:
            page_text = scrape_page_content(url)
            if len(page_text) > 100:  # Consider only if the page has enough content
                combined_scraped_text += page_text + " "
                matched_urls.append(url)

    if not combined_scraped_text.strip():
        return 0.0, [], "No valid content found online."

    # Compute cosine similarity
    vectorizer = TfidfVectorizer().fit_transform([input_text, combined_scraped_text])
    similarity_score = cosine_similarity(vectorizer[0:1], vectorizer[1:2])[0][0]

    return similarity_score, matched_urls, combined_scraped_text[:1000]  # returning preview
