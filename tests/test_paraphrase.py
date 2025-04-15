'''from app.paraphrase_checker import evaluate_text_similarity

doc1 = "Machine learning is a subset of artificial intelligence. It allows systems to learn from data."
doc2 = "Artificial intelligence includes machine learning, which helps computers learn from data."

print("[DEBUG] Running Plagiarism Detection Test...")

results = evaluate_text_similarity(doc1, doc2)

print(f"\n🔍 Cosine Similarity: {results['Cosine Similarity']:.2%}")
print(f"🧠 BLEU Score: {results['BLEU Score']:.4f}")
print(f"📊 ROUGE: {results['ROUGE']}")
print(f"🚨 Plagiarism Percentage: {results['Plagiarism Percentage']:.2f}%")

print("[DEBUG] Test Completed!")
'''

'''
from app.web_scraper import compare_with_web

doc = "AI is transforming industries by automating processes and improving decision-making."
doc = """AI is reshaping industries by automating processes and facilitating smarter decision-making.
With machine learning, systems can analyze data and improve performance over time without being explicitly programmed.
NLP enhances AI's ability to process and generate human language, transforming areas like customer support and content development.
As AI advances, its use cases will grow in domains such as healthcare, financial services, and beyond.
"""


print("\n[DEBUG] Running Web Plagiarism Detection...")
score, urls, preview = compare_with_web(doc)
print(f"\n🔎 Online Similarity Score: {score:.2%}")
print("🧠 Matching URLs:")
for u in urls:
    print(f" - {u}")
print(f"\n📝 Web content sample:\n{preview[:300]}")
'''

from app.web_scraper import compare_with_web

text = "DSA (Data Structures and Algorithms) is the study of organizing data efficiently using data structures like arrays, stacks, and trees, paired with step-by-step procedures (or algorithms) to solve problems effectively."
similarity, urls, preview = compare_with_web(text)

print("\n🔗 URLs Found:")
print(urls)
print("\n📄 Web Content Preview:")
print(preview)
print("\n📈 Similarity Score:", round(similarity * 100, 2), "%")
