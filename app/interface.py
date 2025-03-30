import streamlit as st
from paraphrase_checker import (
    evaluate_text_similarity,
    detect_paraphrased_pairs,
    highlight_paraphrased_pairs,
)
from utils import preprocess_text
import asyncio
import sys
import pdfplumber
from io import BytesIO

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
        file_type = uploaded_file.name.split(".")[-1]

        if file_type == "pdf":
            # Extract text using pdfplumber
            with pdfplumber.open(uploaded_file) as pdf:
                pages = [page.extract_text() for page in pdf.pages]
                content = "\n".join(pages)
                print(f"📚 Extracted text from PDF: {uploaded_file.name}")
                return content

        # Handle text files normally
        elif file_type == "txt":
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

        # Spinner for loading comparison
        with st.spinner("🔎 Analyzing documents..."):
            results = evaluate_text_similarity(doc1_text, doc2_text)
            paraphrased_pairs = detect_paraphrased_pairs(sentences1, sentences2, threshold=0.8)

        st.success("✅ Document comparison complete!")

        # Highlight paraphrased sentences dynamically
        if paraphrased_pairs:
            doc1_highlighted, doc2_highlighted = highlight_paraphrased_pairs(doc1_text, doc2_text, paraphrased_pairs)

            # Display highlighted results using markdown with visible colors
            # Highlighted Doc 1
            st.subheader("📄 Highlighted Document 1:")
            st.markdown(
                f"""
                <div style="background-color:#F5F5F5; padding: 15px; border-radius: 8px; color: black; line-height: 1.6;">
                {doc1_highlighted}
                </div>
                """,
                unsafe_allow_html=True
            )

            # Highlighted Doc 2
            st.subheader("📄 Highlighted Document 2:")
            st.markdown(
                f"""
                <div style="background-color:#F5F5F5; padding: 15px; border-radius: 8px; color: black; line-height: 1.6;">
                {doc2_highlighted}
                </div>
                """,
                unsafe_allow_html=True
            )



            # Add download buttons for highlighted docs
            def generate_download_link(content, filename, label="📥 Download File"):
                """Generate a download button for highlighted text."""
                b64 = BytesIO(content.encode()).getvalue()
                st.download_button(label, data=b64, file_name=filename, mime="text/html")

            generate_download_link(doc1_highlighted, "highlighted_doc1.html", "📥 Download Highlighted Doc 1")
            generate_download_link(doc2_highlighted, "highlighted_doc2.html", "📥 Download Highlighted Doc 2")

        else:
            st.info("✅ No highly similar sentence pairs detected.")

        # Display paraphrased pairs with confidence levels
        def get_confidence_level(score):
            """Return confidence level based on similarity score."""
            if score >= 0.9:
                return "🔥 High Confidence"
            elif score >= 0.75:
                return "⚡ Medium Confidence"
            else:
                return "⚠️ Low Confidence"

        if paraphrased_pairs:
            st.subheader("🔗 Sentence-Level Similarities")
            for s1, s2, score in paraphrased_pairs:
                confidence = get_confidence_level(score)
                st.write(f"➡️ **Doc 1:** {s1}")
                st.write(f"➡️ **Doc 2:** {s2}")
                st.write(f"🔗 **Similarity:** {score:.2%} ({confidence})")
                st.write("---")
        else:
            st.info("✅ No highly similar sentence pairs detected.")

        # Display overall evaluation results
        st.subheader("📊 Evaluation Metrics")
        st.write(f"**Cosine Similarity Score:** {results['Cosine Similarity']:.4f}")
        st.write(f"**BLEU Score:** {results['BLEU Score']:.4f}")
        st.write(f"**Plagiarism Percentage:** {results['Plagiarism Percentage']:.2f}%")

        # Display ROUGE Scores with safety checks
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

        print("✅ Document comparison complete!")
