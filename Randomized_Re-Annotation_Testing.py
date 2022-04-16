import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import re
import os
import codecs
import logging
import warnings; warnings.filterwarnings('ignore')

from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

def tweet_processing(raw_tweet):
    letters_only=re.sub("[^a-zA-Z]"," ",raw_tweet)
    words=letters_only.lower().split()
    stops=set(stopwords.words("english"))
    m_w=[w for w in words if not w in stops]
    return (" ".join(m_w))


# ---------------- MAIN METHOD ----------------- # 


final_dict = {}
final = pd.DataFrame()

# Testing 10 Times
for x in range(0, 10):

    # Load Original DataSet
    df = pd.read_csv("dataset.csv").drop_duplicates()

    hate_speech = df.query('Class=="hate"').copy().reset_index()
    test_hate = hate_speech[:162].copy()
    data_hate = hate_speech[162:3009].copy()

    non_hate_speech = df.query('Class=="none"').copy().reset_index()
    test_non = non_hate_speech[:162].copy()
    data_none = non_hate_speech[162:3009].copy()

    test_data = pd.merge(test_hate, test_non, how='outer').sample(
        frac=1).reset_index()
    data = pd.merge(data_hate, data_none, how='outer').sample(frac=1).reset_index()

    df = pd.merge(data, test_data, how='outer').copy()

    df.isnull().values.any()

    print("Length of original dataframe:", str(len(df)))
    # Training SVM
    X_train, X_test_svm, Y_train, Y_test_svm = train_test_split(df.Tweets, df.Class, test_size=0.2)

    vectorizer=CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,stop_words = None,max_features = 5000)

    train_data_features=vectorizer.fit_transform(X_train)
    train_data_features=train_data_features.toarray()

    test_data_features=vectorizer.transform(X_test_svm)
    test_data_features=test_data_features.toarray()

    #SVM with linear kernel
    clf=svm.SVC(kernel='linear', C=1.0)

    print ("Training SVM: " + str(x + 1))
    clf.fit(train_data_features,Y_train)

    print ("Testing SVM: " + str(x + 1))
    predicted=clf.predict(test_data_features)

    accuracy=np.mean(predicted==Y_test_svm)
    print ("Accuracy: ",accuracy)

    newDict = {"Tweets" : X_test_svm, "Actual" : Y_test_svm, "Predicted" : predicted}

    df2 = pd.DataFrame(newDict).reset_index()

    true_pos = df2.query('Actual == "hate" and Actual == Predicted').copy().reset_index()
    true_neg = df2.query('Actual == "none" and Actual == Predicted').copy().reset_index()
    false_neg = df2.query('Actual == "hate" and Actual != Predicted').copy().reset_index()
    false_pos = df2.query('Actual == "none" and Actual != Predicted').copy().reset_index()

    tp = len(true_pos) / len(df2)
    tn = len(true_neg) / len(df2)
    fn = len(false_neg) / len(df2)
    fp = len(false_pos) / len(df2)
    
    s = "SVM"

    # end of SVM 

    # Average Percentage of false_neg and false_pos
    
    n = int((len(false_neg) + len(false_pos)) / len(df2))

    print("Re-Annotating: " + str(x + 1) + " Number of Re-Annotations: " + str(n) + " : " + str(len(df)))

    # fp_tweets = pd.Series(false_pos.sample(n=n).Tweets)
    # fn_tweets = pd.Series(false_neg.sample(n=n).Tweets)
    copy_df = df.copy()

    # copy_df.query('Tweets in @fn_tweets').replace('hate', 'none')
    copy_df.query('Tweets in @fp_tweets').replace('none', 'hate')
    fileName = "random_reannotated_" + str(x + 1) + "_2.csv"
    copy_df.to_csv(fileName)
    
    # Re-Annotate Data
    r_df = pd.read_csv(fileName).drop(columns=['level_0', 'index', 'Unnamed: 0']).drop_duplicates()

    r_hate_speech = r_df.query('Class=="hate"').copy().reset_index()
    r_test_hate = r_hate_speech[:162].copy()
    r_data_hate = r_hate_speech[162:3009].copy()

    r_non_hate_speech = r_df.query('Class=="none"').copy().reset_index()
    r_test_non = r_non_hate_speech[:162].copy()
    r_data_none = r_non_hate_speech[162:3009].copy()

    r_test_data = pd.merge(r_test_hate, r_test_non, how='outer').sample(frac=1).reset_index()
    r_data = pd.merge(r_data_hate, r_data_none, how='outer').sample(frac=1).reset_index()

    r_df = pd.merge(r_data, r_test_data, how='outer').copy()

    r_df.isnull().values.any()


    print("Length of Re-Annotated DataSet:", str(len(r_df)))
    

    r_X_train, r_X_test_svm, r_Y_train, r_Y_test_svm = train_test_split(r_df.Tweets, r_df.Class, test_size=0.2)

    r_vectorizer=CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,stop_words = None,max_features = 5000)

    r_train_data_features=r_vectorizer.fit_transform(r_X_train)
    r_train_data_features=r_train_data_features.toarray()

    r_test_data_features=r_vectorizer.transform(r_X_test_svm)
    r_test_data_features=r_test_data_features.toarray()

    #SVM with linear kernel
    r_clf=svm.SVC(kernel='linear',C=1.0)

    print ("Training R_SVM: " + str(x + 1))
    r_clf.fit(r_train_data_features,r_Y_train)

    print ("Testing R_SVM: " + str(x + 1))
    r_predicted=clf.predict(r_test_data_features)

    r_accuracy=np.mean(r_predicted==r_Y_test_svm)
    print ("Accuracy: ",r_accuracy)

    r_newDict = {"Tweets" : r_X_test_svm, "Actual" : r_Y_test_svm, "Predicted" : r_predicted}

    r_df2 = pd.DataFrame(r_newDict).reset_index()

    r_true_pos = r_df2.query('Actual == "hate" and Actual == Predicted').copy().reset_index()
    r_true_neg = r_df2.query('Actual == "none" and Actual == Predicted').copy().reset_index()
    r_false_neg = r_df2.query('Actual == "hate" and Actual != Predicted').copy().reset_index()
    r_false_pos = r_df2.query('Actual == "none" and Actual != Predicted').copy().reset_index()

    r_tp = len(r_true_pos) / len(r_df2)
    r_tn = len(r_true_neg) / len(r_df2)
    r_fn = len(r_false_neg) / len(r_df2)
    r_fp = len(r_false_pos) / len(r_df2)
    
    rs = "r_SVM"
    # end of Re-Annotate Data SVM

    iter_dict = {s : {"TP" : tp, "TN" : tn, "FN" : fn, "FP" : fp}, rs : {"TP" : r_tp, "TN" : r_tn, "FN" : r_fn, "FP" : r_fp}}

    print("Trial: " + str(x+1) + str(iter_dict))

    final_dict[x+1] = iter_dict
    
    break

# end of iterative loop
final = pd.DataFrame(final_dict)
final.to_csv("testing.csv")
