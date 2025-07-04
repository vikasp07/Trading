from textblob import TextBlob

def process_youtube_transcript(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    return sentiment_score
