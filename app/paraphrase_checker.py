import numpy as np
from sentence_transformers import SentenceTransformer, util

# Load Sentence Transformer model
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

def get_embeddings(text):
    """Convert text into an embedding."""
    return model.encode([text], convert_to_tensor=True)

def calculate_similarity(doc1, doc2):
    """Compute cosine similarity between two document embeddings."""
    emb1 = get_embeddings(doc1)
    emb2 = get_embeddings(doc2)
    
    similarity_score = util.cos_sim(emb1, emb2).item()  # Extract scalar similarity value
    return similarity_score  # Value between -1 and 1
