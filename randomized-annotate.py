import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

'''


'''

# Loading Data
df = pd.read_csv("dataset.csv").drop_duplicates()

# Equalizing Data Count
hate_df = df.query('Class == "hate"').sample(n=3000).copy().reset_index()
none_df = df.query('Class == "none"').sample(n=3000).copy().reset_index()

df = pd.merge(hate_df, none_df, how="outer").copy()

# DataFrame Stats
hate_per = len(df.query('Class == "hate"'))/ len(df)
none_per = len(df.query('Class == "none"')) / len(df)

print("\nLength of DataFrame:", len(df),"\nPercentage of Hate Tweets:", hate_per, "\nPercentage of non-Hate Tweets:",none_per)

# Split Data for SVM
X_train, X_test, Y_train, Y_test = train_test_split(df.Tweets, df.Class, test_size=0.2)

# Split Data Stats
train_hate_q = Y_train.value_counts()['hate']
train_none_q = Y_train.value_counts()['none']
train_len = len(Y_train)

print("\nLength of Train Split:", train_len, "\nPercentage of Hate Tweets:", train_hate_q / train_len, "\nPercentage of non-Hate Tweets:", train_none_q / train_len)

test_hate_q = Y_test.value_counts()['hate']
test_none_q = Y_test.value_counts()['none']
test_len = len(Y_test)

print("\nLength of Test Split:", test_len, "\nPercentage of Hate Tweets:", test_hate_q / test_len, "\nPercentage of non-Hate Tweets:", test_none_q / test_len)

# Fitting Data to SVM
vectorizer=CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,stop_words = None,max_features = 5000)

train_data_features=vectorizer.fit_transform(X_train)
train_data_features=train_data_features.toarray()

test_data_features=vectorizer.transform(X_test)
test_data_features=test_data_features.toarray()

# SVM with linear kernel
classifier = svm.SVC(kernel='linear', C=1.0)

# Training Classifier
print ("\nTraining SVM...")
classifier.fit(train_data_features,Y_train)

# Testing Classifier
print ("Testing SVM...")
predicted = classifier.predict(test_data_features)

# Show scores
score_svm=precision_recall_fscore_support(Y_test, predicted, average='weighted')
print("\nPrecision:", score_svm[0], "\nRecall:", score_svm[1], "\nF1 Score:", score_svm[2])

# Re-Annotate Y_test Data

test_dict = {"Actual" : Y_test, "Predicted" : predicted}

# print(str(test_dict[0]))

df2 = pd.DataFrame(test_dict)

fn_q = len(df2.query('Actual == "hate" and Actual != Predicted'))
fp_q = len(df2.query('Actual == "none" and Actual != Predicted'))

fn_p = fn_q / len(df2)
fp_p = fp_q / len(df2)

print("\nWill randomly re-annotate: " + str((fn_p * 100)) + "% of hate speech\nWill randomly re-annotate: " +  str((fp_p * 100)) + "% of non-hate speech")


test = pd.DataFrame(Y_test)

test = test.query('Class == "hate"').sample(frac=fn_p).replace("hate", "none")
test = test.query('Class == "none"').sample(frac=fp_p).replace("none", "hate")

Y_test = pd.Series(test['Class'])

test_hate_q = Y_test.value_counts()["hate"]
test_none_q = Y_test.value_counts()["none"]
test_len = len(Y_test)

print("\nLength of Test Split:", test_len, "\nPercentage of Hate Tweets:", test_hate_q / test_len, "\nPercentage of non-Hate Tweets:", test_none_q / test_len)

# Re-Testing Classifier
print ("Re-Testing SVM...")
r_predicted = classifier.predict(test_data_features)

# Show scores
score_svm=precision_recall_fscore_support(Y_test, r_predicted, average='weighted')
print("\nPrecision:", score_svm[0], "\nRecall:", score_svm[1], "\nF1 Score:", score_svm[2])