# import requests

# url = "https://nlp-translation.p.rapidapi.com/v1/translate"

# querystring = {"text":"Hello, world!!","to":"en"}

# headers = {
# 	"X-RapidAPI-Host": "nlp-translation.p.rapidapi.com",
# 	"X-RapidAPI-Key": "03e06ba00fmsh87d23ce2789ef17p1a37fajsn01dd7decb559"
# }

# response = requests.request("GET", url, headers=headers, params=querystring)

# print(response.text)


# from re import I
# import pandas as pd

# df = pd.read_csv("udemydataset/processed_data.csv")

# from langdetect import detect
# def detect_lang(v,i):
#     print(i)
#     try:
#         return detect(v)
#     except Exception:
#         return ''
# df['Language'] = [detect_lang(df['Summary'][x],x) for x in df.index]
# print(df.info())
# df.to_csv("datapro.csv")
# df.dropna(inplace=True)
# df.to_csv("newclf.csv")

import pandas as pd

df = pd.read_csv('newclf.csv')

index_names = df[(len(df['Language']) == 0)].index
print(index_names)
# # drop these given row
# # indexes from dataFrame
# print(len(df))
# df = df[len(df['Language'])]
# df.drop(index_names, inplace = True)
print(df.info())
print(len(df))
