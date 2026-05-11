import pandas as pd

df = pd.read_csv("../DATA/GPT_reviews.csv")

print(df.head())

print("\nColumns:")
print(df.columns)

print("\nDataset Shape:")
print(df.shape)