import streamlit as st
from paraphrase_checker import evaluate_text_similarity
from paraphrase_checker import detect_paraphrased_pairs


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

        st.write("**Processed Sentences (Doc 1):**", sentences1)
        st.write("**Processed Sentences (Doc 2):**", sentences2)

        try:
            embeddings1 = get_embeddings(sentences1)
            embeddings2 = get_embeddings(sentences2)

            overall_similarity, similarity_matrix = calculate_similarity(embeddings1, embeddings2)

            st.subheader(f"🔍 Similarity Score: {overall_similarity:.2%}")

            # Run sentence-level analysis
            paraphrased_pairs = detect_paraphrased_pairs(sentences1, sentences2, threshold=0.8)

            if paraphrased_pairs:
                st.subheader("🔗 Sentence-Level Similarities")
                for s1, s2, score in paraphrased_pairs:
                    st.write(f"➡️ **Doc 1:** {s1}")
                    st.write(f"➡️ **Doc 2:** {s2}")
                    st.write(f"🔗 **Similarity:** {score:.2%}")
                    st.write("---")
            else:
                st.info("No highly similar sentence pairs detected.")

        except ValueError as e:
            st.error(f"Error: {e}")
