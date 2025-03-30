import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import precision_recall_fscore_support

# Load Sentence Transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def get_embeddings(sentences):
    """Convert sentences into embeddings."""
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

def calculate_rouge(reference, candidate):
    """Compute ROUGE scores correctly with valid metric names."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores


def calculate_bleu(reference, candidate):
    """Compute BLEU score between two texts."""
    reference_tokens = [reference.split()]
    candidate_tokens = candidate.split()
    bleu_score = sentence_bleu(reference_tokens, candidate_tokens)
    return bleu_score

def calculate_plagiarism_percentage(similarity_matrix):
    """Estimate the percentage of plagiarism based on similarity scores."""
    threshold = 0.7  # Consider text as plagiarized if similarity > 0.7
    similar_sentences = (similarity_matrix > threshold).sum().item()
    total_sentences = similarity_matrix.numel()
    plagiarism_percentage = (similar_sentences / total_sentences) * 100
    return plagiarism_percentage

def evaluate_text_similarity(text1, text2):
    """Compute all evaluation metrics for text similarity."""
    embeddings1 = get_embeddings([text1])
    embeddings2 = get_embeddings([text2])

    cosine_similarity, similarity_matrix = calculate_similarity(embeddings1, embeddings2)
    rouge_scores = calculate_rouge(text1, text2)
    bleu_score = calculate_bleu(text1, text2)
    plagiarism_percentage = calculate_plagiarism_percentage(similarity_matrix)

    return {
        "Cosine Similarity": cosine_similarity,
        "ROUGE": rouge_scores,
        "BLEU Score": bleu_score,
        "Plagiarism Percentage": plagiarism_percentage
    }

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
