from transformers import pipeline
import json

REVIEWS_FILE = 'reviews.txt'
DATA_FILE = 'data.json'

def compute_sentiments(file):
    reviews = None
    with open(file) as file:
        reviews = [line.strip() for line in file.readlines()]
    if reviews:
        sentiment_pipeline = pipeline("text-classification", model="LiYuan/amazon-review-sentiment-analysis")
        sentiments = sentiment_pipeline(reviews)
        data = []

        for review, sentiment in zip(reviews, sentiments):
            data.append({
                'review': review,
                'star rating': sentiment['label'],
                'score': sentiment['score']
            })
        with open(DATA_FILE, 'w') as data_file:
            data_file.write(json.dumps(data))


if __name__ == '__main__':
    compute_sentiments(REVIEWS_FILE)