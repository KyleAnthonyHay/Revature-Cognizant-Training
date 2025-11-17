import os
import pandas as pd

df  = pd.read_csv("dataset/products.csv")
print(df.head())

# Clean column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("-", "_")
)
# Show cleaned column names 
print(df.head())   