from re import T
from numpy import integer
import pandas as pd
import csv

b_change = {}
r_change = {}
i_change = {}
rvo_change = {}
ivo_change = {}

df = pd.read_csv("results.csv")

for row in df.iterrows():
    # metric strings
    m = row[1][0]
    m_change = m + "-Change"
    print(row)
    print(m)
    print(m_change)

    # set of each result of each trial for a specific metric
    b = [float(row[1][t]) for t in range(1,20,5)]
    r = [float(row[1][t]) for t in range(2,20,5)]
    i = [float(row[1][t]) for t in range(3,20,5)]
    rvo = [float(row[1][t]) for t in range(4,20,5)]
    ivo = [float(row[1][t]) for t in range(5,20,5)]

    # mean average performance
    b_change[m] = sum(b) / len(b)
    r_change[m] = sum(r) / len(r)
    i_change[m] = sum(i) / len(i)
    rvo_change[m] = sum(rvo) / len(rvo)
    ivo_change[m] = sum(ivo) / len(ivo)

    # mean average performance shift
    b_change[m_change] = (sum(b) / len(b)) - (sum(b) / len(b))
    r_change[m_change] = (sum(r) / len(r)) - (sum(b) / len(b))
    i_change[m_change] = (sum(i) / len(i)) - (sum(b) / len(b))
    rvo_change[m_change] = (sum(rvo) / len(rvo)) - (sum(b) / len(b))
    ivo_change[m_change] = (sum(ivo) / len(ivo)) - (sum(b) / len(b))

res = {"Baseline" : b_change, "Randomized" : r_change, "Intelligent" : i_change, "Randomized-Original" : rvo_change, "Intelligent-Original" : ivo_change}
df = pd.DataFrame.from_dict(res)
df.to_csv("fixed.csv")
print(df)
