import streamlit as st
from paraphrase_checker import evaluate_text_similarity, detect_paraphrased_pairs
from utils import preprocess_text
import asyncio
import sys

# 🩹 Fix for event loop error in Python 3.12
if sys.version_info >= (3, 12):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
# End of fix event loop error for Streamlit

st.title("📄 Plagiarism and Paraphrase Detector")

# Upload two documents
uploaded_file1 = st.file_uploader("📥 Upload Document 1", type=["txt", "pdf"])
uploaded_file2 = st.file_uploader("📥 Upload Document 2", type=["txt", "pdf"])


# Read uploaded documents
def read_document(uploaded_file):
    if uploaded_file is not None:
        content = uploaded_file.read().decode("utf-8")
        print(f"📚 Read document: {uploaded_file.name}")
        return content
    return ""


doc1_text = read_document(uploaded_file1)
doc2_text = read_document(uploaded_file2)

# Display text previews
if doc1_text and doc2_text:
    st.subheader("📄 Document 1 Preview:")
    st.text_area("Doc 1", doc1_text, height=150)

    st.subheader("📄 Document 2 Preview:")
    st.text_area("Doc 2", doc2_text, height=150)

    # Preprocess documents for sentence-level similarity
    sentences1 = preprocess_text(doc1_text)
    sentences2 = preprocess_text(doc2_text)
    print(f"🔍 Preprocessed {len(sentences1)} sentences from Doc 1 and {len(sentences2)} sentences from Doc 2.")

    # Compute similarity metrics
    if st.button("🔍 Compare Documents"):
        print("🔎 Starting document comparison...")
        results = evaluate_text_similarity(doc1_text, doc2_text)

        # Run sentence-level paraphrase detection
        paraphrased_pairs = detect_paraphrased_pairs(sentences1, sentences2, threshold=0.8)

        # Display paraphrased pairs
        if paraphrased_pairs:
            st.subheader("🔗 Sentence-Level Similarities")
            for s1, s2, score in paraphrased_pairs:
                st.write(f"➡️ **Doc 1:** {s1}")
                st.write(f"➡️ **Doc 2:** {s2}")
                st.write(f"🔗 **Similarity:** {score:.2%}")
                st.write("---")
        else:
            st.info("✅ No highly similar sentence pairs detected.")

        # Display overall evaluation results
        st.subheader("📊 Evaluation Metrics")
        st.write(f"**Cosine Similarity Score:** {results['Cosine Similarity']:.4f}")
        st.write(f"**BLEU Score:** {results['BLEU Score']:.4f}")
        st.write(f"**Plagiarism Percentage:** {results['Plagiarism Percentage']:.2f}%")

        # Display ROUGE Scores
        st.subheader("📌 ROUGE Scores")
        for key, value in results["ROUGE"].items():
            st.write(
                f"**{key.upper()} Score**: "
                f"Precision={value[0]:.4f}, "
                f"Recall={value[1]:.4f}, "
                f"F1={value[2]:.4f}"
            )
        print("✅ Document comparison complete!")
