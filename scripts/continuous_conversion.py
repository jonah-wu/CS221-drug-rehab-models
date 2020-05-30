import pandas as pd
import numpy as np

print("Importing data")
data = pd.read_csv("500000 teds/update_one_hot_2017_1.csv")

COLUMN_HEADERS = ["age"]

converted_data = pd.DataFrame(np.zeros((data.shape[0], len(COLUMN_HEADERS))), columns=COLUMN_HEADERS)

for i in range(data.shape[0]):
    to_fill = dict.fromkeys(COLUMN_HEADERS, 0)
    row = data.iloc[i]

    #gender
    temp = row["AGE"]
    if temp == 1:
        to_fill["age"] = 13
    elif temp == 2:
        to_fill["age"] = 16
    elif temp == 3:
        to_fill["age"] = 19