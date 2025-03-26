import re

def preprocess_text(text):
    """Clean and split text into sentences."""
    print("[DEBUG] Original Text:", text)

    # Remove special characters and extra spaces
    text = re.sub(r'[^\w\s]', '', text)
    print("[DEBUG] Cleaned Text:", text)

    # Split text into sentences by period
    sentences = text.split('.')
    print("[DEBUG] Split Sentences:", sentences)

    # Strip leading/trailing spaces and remove empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    print("[DEBUG] Final Processed Sentences:", sentences)

    return sentences
