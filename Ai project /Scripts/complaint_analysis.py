import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("../DATA/GPT_reviews.csv")

# Filter negative reviews
negative_reviews = df[df["Rating"] <= 2]

# Complaint categories
categories = {
    "performance": [
        "glitch",
        "slow",
        "lag",
        "crash",
        "overheat",
        "battery"
    ],

    "pricing_limits": [
        "limit",
        "wait",
        "subscription",
        "premium",
        "paid"
    ],

    "privacy": [
        "recording",
        "listening",
        "privacy",
        "spy"
    ],

    "accuracy": [
        "wrong",
        "bad answer",
        "incorrect",
        "not proper"
    ],

    "censorship": [
        "warning",
        "restricted",
        "terminated",
        "blocked"
    ],

    "memory": [
        "remember",
        "memory",
        "forget"
    ],

    "server_issues": [
        "server",
        "error",
        "not working"
    ]
}

# Count complaints
category_counts = {}

for category, keywords in categories.items():

    count = 0

    for review in negative_reviews["Comment"].astype(str):

        review_lower = review.lower()

        if any(keyword in review_lower for keyword in keywords):
            count += 1

    category_counts[category] = count

# Convert to dataframe
results_df = pd.DataFrame({
    "Category": category_counts.keys(),
    "Count": category_counts.values()
})

# Sort results
results_df = results_df.sort_values(by="Count", ascending=False)

# Print results
print(results_df)

# Create chart
plt.figure(figsize=(10,6))

plt.bar(results_df["Category"], results_df["Count"])

plt.title("Top AI User Complaints")
plt.xlabel("Complaint Category")
plt.ylabel("Number of Complaints")

plt.xticks(rotation=20)

plt.tight_layout()

plt.show()