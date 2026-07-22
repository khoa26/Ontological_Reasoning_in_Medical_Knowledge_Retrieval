import pandas as pd

df = pd.read_parquet("datasets/test-00000-of-00001.parquet")
print(df.head())