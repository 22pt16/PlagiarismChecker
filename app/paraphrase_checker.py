import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
import re

# Load Sentence Transformer model
print("🔄 Loading Sentence Transformer model...")
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
print("✅ Model loaded successfully!")


def get_embeddings(sentences):
    """Convert sentences into embeddings."""
    print(f"📚 Getting embeddings for {len(sentences)} sentences...")
    if not sentences or len(sentences) == 0:
        raise ValueError("❌ No valid sentences provided for embedding.")

    embeddings = model.encode(sentences, convert_to_tensor=True)
    print("✅ Embeddings generated successfully.")
    return embeddings


def calculate_similarity(embeddings1, embeddings2):
    """Compute cosine similarity between two sets of sentence embeddings."""
    print("🔍 Calculating cosine similarity...")
    if not isinstance(embeddings1, torch.Tensor) or not isinstance(embeddings2, torch.Tensor):
        raise ValueError("❌ Embeddings must be PyTorch tensors.")

    similarity_matrix = util.pytorch_cos_sim(embeddings1, embeddings2)
    overall_similarity = similarity_matrix.mean().item()
    print(f"✅ Cosine similarity calculated: {overall_similarity:.4f}")
    return overall_similarity, similarity_matrix


def calculate_rouge(reference, candidate):
    """Compute ROUGE scores correctly with valid metric names."""
    print("📊 Calculating ROUGE scores...")
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    print("✅ ROUGE scores calculated successfully.")
    return scores



def calculate_plagiarism_percentage(similarity_matrix):
    """Estimate the percentage of plagiarism based on similarity scores."""
    print("🔎 Calculating plagiarism percentage...")
    threshold = 0.7  # Consider text as plagiarized if similarity > 0.7
    similar_sentences = (similarity_matrix > threshold).sum().item()
    total_sentences = similarity_matrix.numel()
    plagiarism_percentage = (similar_sentences / total_sentences) * 100
    print(f"✅ Plagiarism percentage: {plagiarism_percentage:.2f}%")
    return plagiarism_percentage


def evaluate_text_similarity(text1, text2):
    """Compute all evaluation metrics for text similarity."""
    print("🔎 Evaluating text similarity...")
    embeddings1 = get_embeddings([text1])
    embeddings2 = get_embeddings([text2])

    cosine_similarity, similarity_matrix = calculate_similarity(embeddings1, embeddings2)
    rouge_scores = calculate_rouge(text1, text2)
    plagiarism_percentage = calculate_plagiarism_percentage(similarity_matrix)

    print("✅ Evaluation completed successfully!")
    return {
        "Cosine Similarity": cosine_similarity,
        "ROUGE": rouge_scores,
        "Plagiarism Percentage": plagiarism_percentage
    }

def detect_paraphrased_pairs(sentences1, sentences2, threshold=0.8):
    """Detect and return paraphrased sentence pairs with similarity above the threshold."""
    print("🔎 Detecting paraphrased sentence pairs...")
    embeddings1 = get_embeddings(sentences1)
    embeddings2 = get_embeddings(sentences2)

    _, similarity_matrix = calculate_similarity(embeddings1, embeddings2)

    paraphrased_pairs = []
    for i, row in enumerate(similarity_matrix):
        for j, score in enumerate(row):
            if score.item() > threshold:
                paraphrased_pairs.append(
                    (sentences1[i], sentences2[j], score.item())
                )
    print(f"✅ Found {len(paraphrased_pairs)} paraphrased pairs.")
    return paraphrased_pairs

def highlight_paraphrased_pairs(doc1, doc2, paraphrased_pairs):
    """Highlight paraphrased sentence pairs correctly in both documents."""
    for s1, s2, score in paraphrased_pairs:
        s1_escaped = re.escape(s1)
        s2_escaped = re.escape(s2)

        # Wrap only the paraphrased part with color
        doc1 = re.sub(
            s1_escaped,
            f"<span style='background-color: #FFD700; color: black; padding: 2px 5px; border-radius: 5px;'>{s1}</span>",
            doc1,
            flags=re.IGNORECASE
        )
        doc2 = re.sub(
            s2_escaped,
            f"<span style='background-color: #FFD700; color: black; padding: 2px 5px; border-radius: 5px;'>{s2}</span>",
            doc2,
            flags=re.IGNORECASE
        )

    return doc1, doc2

