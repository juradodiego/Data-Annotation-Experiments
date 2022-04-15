import pandas as pd

ES = []
EN = []

read_file = open("es-en.train.txt", "r")

for line in read_file:
    pairs = line.split(' ')
    pairs[1] = pairs[1].replace("\n","")
    ES.append(pairs[0])
    EN.append(pairs[1])


dict = {'ES':ES,
        'EN':EN
        }
df = pd.DataFrame(dict)

df.to_csv("es-en-word-pairs.csv")