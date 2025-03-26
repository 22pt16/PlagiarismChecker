from app.paraphrase_checker import detect_paraphrases

doc1 = "AI is transforming the world. Technology is advancing rapidly."
doc2 = "Artificial intelligence is changing the world. The pace of technological advancement is fast."

print("[DEBUG] Running Paraphrase Detection Test...")

# Run paraphrase detection
paraphrased_pairs, overall_similarity = detect_paraphrases(doc1, doc2)

print(f"\n✅ Overall Similarity: {overall_similarity:.2%}")
for s1, s2, score in paraphrased_pairs:
    print(f"➡️ Doc 1: {s1}\n➡️ Doc 2: {s2}\n🔗 Similarity: {score:.2%}\n---")

print("[DEBUG] Test Completed!")

