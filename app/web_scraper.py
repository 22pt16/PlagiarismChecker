# app/web_scraper.py
from keybert import KeyBERT
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import urllib.parse

kw_model = KeyBERT()

def extract_keywords(text, top_n=5):
    
    from sklearn.feature_extraction.text import CountVectorizer

    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(text, 
                                         keyphrase_ngram_range=(1, 2), 
                                         stop_words='english', 
                                         use_maxsum=True, 
                                         nr_candidates=20, 
                                         top_n=top_n)
    return [kw[0] for kw in keywords]




def fetch_search_results(query):
    headers = {"User-Agent": "Mozilla/5.0"}
    search_url = f"https://www.google.com/search?q=wikipedia+{urllib.parse.quote_plus(query)}"
    print("🔗 Searching:", search_url)

    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    links = []

    # Try getting the main links
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "wikipedia.org" in href and "/url?q=" in href:
            # Clean the /url?q= prefix
            clean_url = href.split("/url?q=")[1].split("&")[0]
            links.append(clean_url)

    return links[:5]




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
        print("✅ URLs found:", urls)
        for url in urls:
            page_text = scrape_page_content(url)
            if len(page_text) > 25:  # Consider only if the page has enough content
                combined_scraped_text += page_text + " "
                matched_urls.append(url)

    if not combined_scraped_text.strip():
        return keywords, 0.0, [], "No valid content found online."

    # Compute cosine similarity
    vectorizer = TfidfVectorizer().fit_transform([input_text, combined_scraped_text])
    similarity_score = cosine_similarity(vectorizer[0:1], vectorizer[1:2])[0][0]

    return keywords, similarity_score, matched_urls, combined_scraped_text[:1000]  # returning preview
