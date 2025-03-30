import re

def preprocess_text(text):
    """Clean and split text into sentences."""
    if not text or text.strip() == "":
        return ["Empty document"]  # Ensures non-empty input

    # Remove special characters (except periods) and normalize whitespace
    text = re.sub(r'[^\w\s.]', '', text).strip()
    
    # Split text into sentences
    sentences = re.split(r'\.\s*', text)  # Split by period and space
    
    # Remove empty elements and strip spaces
    sentences = [s.strip() for s in sentences if s.strip()]

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
