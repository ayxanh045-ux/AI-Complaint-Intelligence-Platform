import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

# Load dataset
df = pd.read_csv("DATA/GPT_reviews.csv")

# Filter negative reviews
negative_reviews = df[df["Rating"] <= 2]

# Combine text
text = " ".join(
    negative_reviews["Comment"].astype(str)
)

# Custom stopwords
custom_stopwords = set(STOPWORDS)

custom_stopwords.update([
    "app",
    "chatgpt",
    "ai",
    "use",
    "using",
    "work",
    "answer",
    "please",
    "would",
    "even",
    "one",
    "get",
    "now",
    "can",
    "will",
    "really"
])

# Generate cleaner word cloud
wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color="white",
    stopwords=custom_stopwords
).generate(text)

# Display
plt.figure(figsize=(14,7))

plt.imshow(wordcloud)

plt.axis("off")

plt.title("Most Common AI Complaints")

plt.show()