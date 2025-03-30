import streamlit as st
from paraphrase_checker import calculate_similarity

st.title("Plagiarism Checker")

# Input text areas for two documents
doc1 = st.text_area("Enter Document 1:", height=200)
doc2 = st.text_area("Enter Document 2:", height=200)

if st.button("Check for Plagiarism"):
    if doc1.strip() and doc2.strip():
        similarity_score = calculate_similarity(doc1, doc2)
        st.write(f"**Similarity Score:** {similarity_score:.2%}")

        if similarity_score > 0.8:
            st.error("⚠️ High similarity detected! Possible plagiarism.")
        elif similarity_score > 0.5:
            st.warning("⚠️ Moderate similarity detected. Some content overlap found.")
        else:
            st.success("✅ Low similarity. No significant plagiarism detected.")
    else:
        st.error("Please enter text in both documents before checking.")
