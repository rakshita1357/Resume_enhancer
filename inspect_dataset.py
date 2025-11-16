import pandas as pd

df = pd.read_csv("data/overall_ds.csv", encoding="ISO-8859-1")

print(df.columns)
print(df.head(10))
print("Rows:", len(df))
print("Null values:\n", df.isnull().sum())
