from re import T
from numpy import integer
import pandas as pd
import csv

b_change = {}
r_change = {}
i_change = {}

df = pd.read_csv("results.csv")

for row in df.iterrows():
    # metric strings
    m = row[1][0]
    m_change = m + "-Change"

    # set of each result of each trial for a specific metric
    b = [float(row[1][t]) for t in range(1,46,3)]
    r = [float(row[1][t]) for t in range(2,46,3)]
    i = [float(row[1][t]) for t in range(3,46,3)]
    
    # mean average performance 
    b_change[m] = sum(b) / len(b)
    r_change[m] = sum(r) / len(r)
    i_change[m] = sum(i) / len(i)

    # mean average performance shift
    b_change[m_change] = (sum(b) / len(b)) - (sum(b) / len(b))
    r_change[m_change] = (sum(r) / len(r)) - (sum(b) / len(b))
    i_change[m_change] = (sum(i) / len(i)) - (sum(b) / len(b))

res = {"Baseline" : b_change, "Randomized" : r_change, "Intelligent" : i_change}
df = pd.DataFrame.from_dict(res)
print(df)
