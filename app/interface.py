import streamlit as st
from paraphrase_checker import get_embeddings, calculate_similarity
from paraphrase_checker import detect_paraphrased_pairs

from utils import preprocess_text

st.title("Plagiarism Checker")

uploaded_file1 = st.file_uploader("Upload First Document", type=["txt"])
uploaded_file2 = st.file_uploader("Upload Second Document", type=["txt"])

if uploaded_file1 and uploaded_file2:
    doc1 = uploaded_file1.read().decode("utf-8").strip()
    doc2 = uploaded_file2.read().decode("utf-8").strip()

    if not doc1 or not doc2:
        st.error("❌ One or both documents are empty. Please upload valid text files.")
    else:
        st.subheader("Uploaded Documents:")
        st.write("**Document 1 Preview:**", doc1[:500])
        st.write("**Document 2 Preview:**", doc2[:500])  

        # Preprocess text
        sentences1 = preprocess_text(doc1)
        sentences2 = preprocess_text(doc2)

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
