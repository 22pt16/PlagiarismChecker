import re

from paraphrase_checker import calculate_similarity, get_embeddings

def preprocess_text(text):
    """Clean and split text into sentences."""
    print("🔄 Preprocessing text...")
    if not text or text.strip() == "":
        print("⚠️ Empty or invalid document provided!")
        return ["Empty document"]

    # Remove special characters (except periods) and normalize whitespace
    text = re.sub(r'[^\w\s.]', '', text).strip()
    sentences = re.split(r'\.\s*', text)  # Split by period and space

    sentences = [s.strip() for s in sentences if s.strip()]
    print(f"✅ Preprocessed {len(sentences)} sentences.")
    return sentences if sentences else ["No meaningful content"]

def detect_paraphrased_pairs(sentences1, sentences2, threshold=0.8):
    """Detect and return paraphrased sentence pairs with similarity above the threshold."""
    embeddings1 = get_embeddings(sentences1)
    embeddings2 = get_embeddings(sentences2)

    # Calculate sentence-level similarity matrix
    _, similarity_matrix = calculate_similarity(embeddings1, embeddings2)

    paraphrased_pairs = []
    for i, row in enumerate(similarity_matrix):
        for j, score in enumerate(row):
            if score.item() > threshold:
                paraphrased_pairs.append(
                    (sentences1[i], sentences2[j], score.item())
                )

    return paraphrased_pairs
