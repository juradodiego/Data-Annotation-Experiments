import pandas as pd

df=pd.read_csv("final_dataset.csv")

df = df.replace(['sexism', 'racism'],'hate')

df.to_csv("dataset.csv")