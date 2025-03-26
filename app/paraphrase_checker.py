import numpy as np
from sentence_transformers import SentenceTransformer, util
from app.utils import preprocess_text

# Load Sentence Transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def get_embeddings(sentences):
    """Convert sentences into embeddings."""
    print(f"[DEBUG] Generating embeddings for {len(sentences)} sentences...")
    embeddings = model.encode(sentences, convert_to_tensor=True)
    print("[DEBUG] Embeddings generated successfully!")
    return embeddings

def calculate_similarity(embeddings1, embeddings2):
    """Compute cosine similarity between two sets of embeddings."""
    print("[DEBUG] Calculating cosine similarity between embeddings...")
    similarity_matrix = util.cos_sim(embeddings1, embeddings2)
    print("[DEBUG] Similarity matrix shape:", similarity_matrix.shape)
    return similarity_matrix

def detect_paraphrases(doc1, doc2, threshold=0.8):
    """Compare sentences from two documents and detect paraphrases."""
    print("[DEBUG] Starting paraphrase detection...")
    sentences1 = preprocess_text(doc1)
    sentences2 = preprocess_text(doc2)

    embeddings1 = get_embeddings(sentences1)
    embeddings2 = get_embeddings(sentences2)

    similarity_matrix = calculate_similarity(embeddings1, embeddings2)

    paraphrased_pairs = []
    for i, row in enumerate(similarity_matrix):
        for j, score in enumerate(row):
            if score > threshold:
                print(f"[DEBUG] Paraphrase detected: '{sentences1[i]}' ↔ '{sentences2[j]}' | Score: {score:.2%}")
                paraphrased_pairs.append((sentences1[i], sentences2[j], float(score)))
    
    overall_similarity = np.mean(similarity_matrix.numpy())
    print(f"[DEBUG] Overall Similarity: {overall_similarity:.2%}")
    
    return paraphrased_pairs, overall_similarity
