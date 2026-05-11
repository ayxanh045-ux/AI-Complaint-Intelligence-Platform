import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load dataset
df = pd.read_csv("DATA/GPT_reviews.csv")

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to calculate sentiment
def get_sentiment(text):

    score = analyzer.polarity_scores(str(text))

    return score["compound"]

# Apply sentiment analysis
df["sentiment_score"] = df["Comment"].apply(get_sentiment)

# Label sentiment
def label_sentiment(score):

    if score >= 0.05:
        return "Positive"

    elif score <= -0.05:
        return "Negative"

    else:
        return "Neutral"

df["sentiment_label"] = df["sentiment_score"].apply(label_sentiment)

# Count sentiments
sentiment_counts = df["sentiment_label"].value_counts()

print(sentiment_counts)

# Create chart
plt.figure(figsize=(8,6))

plt.bar(
    sentiment_counts.index,
    sentiment_counts.values
)

plt.title("AI Review Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")

plt.tight_layout()

plt.show()