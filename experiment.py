'''
@author     Diego Jurado
@project    CS1699 / CS 1951 Term Research Project
@title      Compensating for Annotation Bias via Re-Annotation Methods
'''

# Imports
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

''' ------ HELPER METHODS ------ '''

# Data Processing
# params: String file
# return: DataFrame df
def import_data(file):
    df = pd.read_csv(file).drop_duplicates()
    # randomly sample 3000 hate speech
    hate = df.query('Class == "hate"').sample(n=3000).copy() 
    # randomly sample 3000 non-hate speech
    non_hate = df.query('Class == "none"').sample(n=3000).copy()
    # merge hate and non-hate
    df = pd.merge(hate, non_hate, how='outer').copy()

    return df

# Data Set Information Reporting
# params: DataFrame df
def data_stat(df):
    length = len(df)
    h_length = len(df.query('Class == "hate"'))
    n_length = len(df.query('Class == "none"'))
    print("\nNumber of samples:", length,"\nNumber of Hate Samples:", h_length, "\nNumber of non-Hate Samples:", n_length)

# Randomized Re-Annotation Sampling
# params: Dataframe df, float fn, float fp
# return: DataFrame rdf
def rand_annot_samp(df, fn, fp):

    rdf = df.copy()
    fn_tweets = df.query('Class == "hate"').sample(frac=fn).Tweets.tolist()
    fp_tweets = df.query('Class == "none"').sample(frac=fp).Tweets.copy()

    for t in fn_tweets:
        rdf.loc[df["Tweets"] == t, "Class"] = "hate"
    for t in fp_tweets:
        rdf.loc[df["Tweets"] == t, "Class"] = "none"

    return rdf

''' TO-DO '''
# Intelligent Re-Annotation Sampling
# params: Dataframe df, float fn, float fp
# return: DataFrame idf
def inte_annot_samp(df):

    idf = df.copy()
    fn_tweets = []
    fp_tweets = []

    for t in fn_tweets:
        idf.loc[df["Tweets"] == t, "Class"] = "hate"
    data_stat(rdf)
    for t in fp_tweets:
        idf.loc[df["Tweets"] == t, "Class"] = "none"
    data_stat(rdf)

    return idf

''' ------ MAIN METHOD ------ '''

for trial in range(1,6):
    print("\nTrial:", trial)

    # ------ Baseline ------ #

    # Data Processing
    df = import_data("dataset.csv")

    # Data Set Info
    data_stat(df)

    # Data Split
    X_train, X_test, Y_train, Y_test = train_test_split(df.Tweets, df.Class, test_size=0.2)

    vectorizer = CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,stop_words = None,max_features = 5000)

    train_data_features = vectorizer.fit_transform(X_train)
    train_data_features = train_data_features.toarray()

    test_data_features = vectorizer.transform(X_test)
    test_data_features = test_data_features.toarray()

    # Classifier
    clf=svm.SVC(kernel='linear', C=1.0)

    print ("\nTraining Baseline SVM")
    clf.fit(train_data_features,Y_train)

    print ("\nTesting Baseline SVM")
    predicted=clf.predict(test_data_features)

    accuracy=np.mean(predicted==Y_test)

    # TO-DO: Format Accuracy Str
    print ("\nBaseline Accuracy: ",accuracy) 

    t_dict = {"Tweets" : X_test, "Actual" : Y_test, "Predicted" : predicted}
    df2 = pd.DataFrame(t_dict).reset_index()

    fn = len(df2.query('Actual == "hate" and Actual != Predicted')) / len(df2)
    fp = len(df2.query('Actual == "none" and Actual != Predicted')) / len(df2)

    # ------ Random Sampling ------ #

    # Data Processing
    rdf = rand_annot_samp(df, fn, fp)

    # Data Set Info
    data_stat(rdf)

    # Data Split
    X_train, X_test, Y_train, Y_test = train_test_split(rdf.Tweets, rdf.Class, test_size=0.2)

    train_data_features = vectorizer.fit_transform(X_train)
    train_data_features = train_data_features.toarray()

    test_data_features = vectorizer.transform(X_test)
    test_data_features = test_data_features.toarray()

    # Classifier
    r_clf = svm.SVC(kernel='linear', C=1.0)

    print("\nTraining Random Sampling SVM")
    r_clf.fit(train_data_features,Y_train)

    print("\nTesting Random Sampling SVM")
    r_predicted = r_clf.predict(test_data_features)

    r_accuracy=np.mean(r_predicted==Y_test)

    # TO-DO: Format Accuracy Str
    print ("\nRandom Accuracy: ",r_accuracy) 
   

    # ------ Intelligent Sampling ------ #

    # Data Processing
    idf = inte_annot_samp(df, fn, fp)

    # Data Set Info
    data_stat(idf)

    # Data Split
    X_train, X_test, Y_train, Y_test = train_test_split(idf.Tweets, idf.Class, test_size=0.2)

    train_data_features = vectorizer.fit_transform(X_train)
    train_data_features = train_data_features.toarray()

    test_data_features = vectorizer.transform(X_test)
    test_data_features = test_data_features.toarray()

    # Classifier
    i_clf = svm.SVC(kernel='linear', C=1.0)

    print("\nTraining Intelligent Sampling SVM")
    i_clf.predict(test_data_features)

    print("\nTesting Intelligent Sampling SVM")
    i_predicted = clf.predict(test_data_features)

    i_accuracy = np.mean(i_predicted==Y_test)

    # TO-DO: Format Accuracy Str
    print ("\nIntelligent Accuracy: ",i_accuracy)  
    