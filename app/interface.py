import streamlit as st
import asyncio
import sys
import requests
import pdfplumber
from io import BytesIO
from bs4 import BeautifulSoup

from paraphrase_checker import (
    evaluate_text_similarity,
    detect_paraphrased_pairs,
    highlight_paraphrased_pairs,
)
from web_scraper import compare_with_web
from utils import preprocess_text


# -------------------- Environment Fix for Python 3.12 --------------------
if sys.version_info >= (3, 12):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# -------------------- Helper Functions --------------------

def extract_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()
        return ' '.join(soup.stripped_strings)
    except Exception as e:
        st.error(f"❌ Error fetching or parsing the URL: {e}")
        return ""


def read_document(uploaded_file):
    if uploaded_file:
        file_type = uploaded_file.name.split(".")[-1].lower()

        if file_type == "pdf":
            with pdfplumber.open(uploaded_file) as pdf:
                return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

        elif file_type == "txt":
            return uploaded_file.read().decode("utf-8")

    return ""


def get_confidence_level(score):
    if score >= 0.9:
        return "🔥 High Confidence"
    elif score >= 0.75:
        return "⚡ Medium Confidence"
    else:
        return "⚠️ Low Confidence"


def display_paraphrased_pairs(pairs):
    st.subheader("🔗 Sentence-Level Similarities")
    for s1, s2, score in pairs:
        confidence = get_confidence_level(score)
        st.write(f"➡️ **Doc 1:** {s1}")
        st.write(f"➡️ **Doc 2:** {s2}")
        st.write(f"🔗 **Similarity:** {score:.2%} ({confidence})")
        st.write("---")


def display_evaluation_results(results):
    st.subheader("📊 Evaluation Metrics")
    st.write(f"**Cosine Similarity Score:** {results['Cosine Similarity']:.4f}")
    st.write(f"**Plagiarism Percentage:** {results['Plagiarism Percentage']:.2f}%")

    st.subheader("📌 ROUGE Scores")
    if results["ROUGE"]:
        for key, value in results["ROUGE"].items():
            if value:
                st.write(
                    f"**{key.upper()} Score**: "
                    f"Precision={value[0]:.4f}, "
                    f"Recall={value[1]:.4f}, "
                    f"F1={value[2]:.4f}"
                )
    else:
        st.warning("⚠️ No valid ROUGE scores generated.")


def display_web_results(doc_text, label="Document"):
    with st.spinner(f"🌐 Searching the web for content similar to {label}..."):
        similarity_score, matched_urls, preview = compare_with_web(doc_text)

    # Debug line (you can remove it later)
    print(f"DEBUG - Web Similarity Results for {label}:", similarity_score, matched_urls, preview)

    st.markdown(f"### 🌐 Web Results for {label}")
    if matched_urls:
        st.markdown(f"**Similarity Score with Online Sources:** `{similarity_score:.2%}`")
        st.markdown("---")
        for i, url in enumerate(matched_urls, 1):
            st.markdown(f"**{i}.** [{url}]({url})")
        st.markdown("---")
        
    else:
        st.info(f"No similar content found online for **{label}**.")




# -------------------- Streamlit App --------------------

st.title("📄 Plagiarism and Paraphrase Detector")

uploaded_file1 = st.file_uploader("📥 Upload Document 1", type=["txt", "pdf"])
uploaded_file2 = st.file_uploader("📥 Upload Document 2", type=["txt", "pdf"])

doc1_text = read_document(uploaded_file1)
doc2_text = read_document(uploaded_file2)

if doc1_text and doc2_text:
    st.subheader("📄 Document 1 Preview:")
    st.text_area("Doc 1", doc1_text, height=150)

    st.subheader("📄 Document 2 Preview:")
    st.text_area("Doc 2", doc2_text, height=150)

    sentences1 = preprocess_text(doc1_text)
    sentences2 = preprocess_text(doc2_text)

    if st.button("🔍 Compare Documents"):
        with st.spinner("🔎 Analyzing documents..."):
            results = evaluate_text_similarity(doc1_text, doc2_text)
            paraphrased_pairs = detect_paraphrased_pairs(sentences1, sentences2, threshold=0.8)

        st.success("✅ Document comparison complete!")

        if paraphrased_pairs:
            doc1_highlighted, doc2_highlighted = highlight_paraphrased_pairs(doc1_text, doc2_text, paraphrased_pairs)
            display_paraphrased_pairs(paraphrased_pairs)
        else:
            st.info("✅ No highly similar sentence pairs detected.")

        display_evaluation_results(results)

        # ---- Web Plagiarism Results in Sidebar ----
        with st.sidebar.expander("🌐 Web Plagiarism Check for Documents", expanded=False):
            display_web_results(doc1_text, label="Document 1")
            display_web_results(doc2_text, label="Document 2")

        print("✅ Document Parsed")

            
