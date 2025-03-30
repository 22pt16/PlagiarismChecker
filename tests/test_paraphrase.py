from app.paraphrase_checker import check_plagiarism

doc1 = "Machine learning is a subset of artificial intelligence. It allows systems to learn from data."
doc2 = "Artificial intelligence includes machine learning, which helps computers learn from data."

print("[DEBUG] Running Plagiarism Detection Test...")

similarity_score, plagiarism_detected = check_plagiarism(doc1, doc2)

print(f"\n🔍 Similarity Score: {similarity_score:.2%}")
print("🚨 Plagiarism Detected!" if plagiarism_detected else "✅ No significant plagiarism detected.")

print("[DEBUG] Test Completed!")
