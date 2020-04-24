import numpy as np
import pandas as pd


df = pd.read_csv("TEDS-A.csv")
df.head()

for col in df.columns:
	print(col)
