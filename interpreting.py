from numpy import integer
import pandas as pd

df = pd.read_csv("results-15.csv")

avg_random_acc_change = 0
avg_intelligent_acc_change = 0

b_change = []
r_change = []
i_change = []
for trial in range(1,16):
    bl = df.iloc[0][trial]
    rand = df.iloc[1][trial]
    inte = df.iloc[2][trial]

    b_change.append((bl-bl) * 100)
    r_change.append((rand-bl) * 100)
    i_change.append((inte-bl) * 100)

print("Avg Baseline Change: " + str(sum(b_change) / len(b_change)) + "%")
print("Avg Random Change: " + str(sum(r_change) / len(r_change)) + "%")
print("Avg Intelligent Change: " + str(sum(i_change) / len(i_change)) + "%")