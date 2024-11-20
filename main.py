import nltk
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy

nltk.download("movie_reviews")

def extract_features(words):
    return {word: True for word in words}
    
pos_reviews = [(extract_features(movie_reviews.words(fileid)), 'pos') for fileid in movie_reviews.fileids('pos')]
neg_reviews = [(extract_features(movie_reviews.words(fileid)), 'neg') for fileid in movie_reviews.fileids('neg')]

# Combine and shuffle the data
reviews = pos_reviews + neg_reviews

# Shuffle the data
import random
random.shuffle(reviews)

# Split data: 80% for training, 20% for testing
split_index = int(0.8 * len(reviews))
train_data, test_data = reviews[:split_index], reviews[split_index:]
# Train the classifier
classifier = NaiveBayesClassifier.train(train_data)
# Evaluate the classifier
print("Accuracy:", accuracy(classifier, test_data))

# Define a function to classify new text
def classify_text(text):
    words = text.split()
    features = extract_features(words)
    return classifier.classify(features)

# Test with a custom sentence
text = input("Enter text: ")
print(classify_text(text))
