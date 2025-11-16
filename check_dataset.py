import pandas as pd

df = pd.read_csv("data/overall_ds.csv", encoding="ISO-8859-1")
print(df.head())
print("Rows:", len(df))
