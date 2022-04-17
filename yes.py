import pandas as pd

df = pd.read_csv("udemydataset/newclf-newclf.csv")
df["Description"] = df.apply(lambda row: row.Title + row.Summary, axis=1)
df.to_csv("udemydataset/finaldataset.csv")