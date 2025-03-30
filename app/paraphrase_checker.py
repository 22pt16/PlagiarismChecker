import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

# Load model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def get_embeddings(sentences):
    """Convert sentences into embeddings, ensuring non-empty input."""
    if not sentences or len(sentences) == 0:
        raise ValueError("No valid sentences provided for embedding.")

    return model.encode(sentences, convert_to_tensor=True)


def calculate_similarity(embeddings1, embeddings2):
    """Compute cosine similarity between two sets of sentence embeddings."""
    if not isinstance(embeddings1, torch.Tensor) or not isinstance(embeddings2, torch.Tensor):
        raise ValueError("Embeddings must be PyTorch tensors.")

    similarity_matrix = util.pytorch_cos_sim(embeddings1, embeddings2)

    # Compute mean similarity (overall document similarity)
    overall_similarity = similarity_matrix.mean().item()
    return overall_similarity, similarity_matrix

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
