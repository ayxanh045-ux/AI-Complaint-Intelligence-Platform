import pandas as pd
import requests
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("OPENROUTER_API_KEY")

# Load dataset
df = pd.read_csv("DATA/GPT_reviews.csv")

# Filter negative reviews
negative_reviews = df[df["Rating"] <= 2]

# Take sample complaints
sample_complaints = (
    negative_reviews["Comment"]
    .dropna()
    .head(10)
)

# Combine complaints into one text block
complaints_text = "\n".join(sample_complaints)

# Send request to OpenRouter
response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",

    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    },

    json={
        "model": "openai/gpt-3.5-turbo",

        "messages": [

            {
                "role": "system",
                "content": (
                    "You are a senior data analyst "
                    "analyzing AI product complaints."
                )
            },

            {
                "role": "user",
                "content": f"""
Analyze these user complaints about an AI application.

Identify:
- main complaint themes
- emotional patterns
- recurring frustrations
- business insights

Complaints:

{complaints_text}
"""
            }
        ],

        "max_tokens": 300
    }
)

# Convert response to JSON
result = response.json()

# Print AI-generated summary
print("\nAI INSIGHT SUMMARY:\n")

print(
    result["choices"][0]["message"]["content"]
)