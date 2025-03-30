import streamlit as st
from paraphrase_checker import evaluate_text_similarity

st.title("📄 Plagiarism and Paraphrase Detector")

# Upload two documents
uploaded_file1 = st.file_uploader("Upload Document 1", type=["txt", "pdf"])
uploaded_file2 = st.file_uploader("Upload Document 2", type=["txt", "pdf"])

# Read uploaded documents
def read_document(uploaded_file):
    if uploaded_file is not None:
        return uploaded_file.read().decode("utf-8")
    return ""

doc1_text = read_document(uploaded_file1)
doc2_text = read_document(uploaded_file2)

# Display text previews
if doc1_text and doc2_text:
    st.subheader("Document 1 Preview:")
    st.text_area("Doc 1", doc1_text, height=150)

    st.subheader("Document 2 Preview:")
    st.text_area("Doc 2", doc2_text, height=150)

    # Compute similarity metrics
    if st.button("Compare Documents"):
        results = evaluate_text_similarity(doc1_text, doc2_text)

        st.subheader("📊 Evaluation Metrics")
        st.write(f"**Cosine Similarity Score:** {results['Cosine Similarity']:.4f}")
        st.write(f"**BLEU Score:** {results['BLEU Score']:.4f}")
        st.write(f"**Plagiarism Percentage:** {results['Plagiarism Percentage']:.2f}%")

        st.subheader("📌 ROUGE Scores")
        for key, value in results["ROUGE"].items():
            st.write(f"**{key.upper()} Score**: Precision={value.precision:.4f}, Recall={value.recall:.4f}, F1={value.fmeasure:.4f}")
