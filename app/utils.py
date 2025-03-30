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
