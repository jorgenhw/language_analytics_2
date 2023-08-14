"""
#### IMPORTS ####
"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


"""
#### FUNCTIONS ####
"""


# Function for feature extraction and vectorize data (with TF-IDF)
def tfidf_vectorize_data(X_train, X_test, ngram_range = (1,2), lowercase =  True, max_df = 0.95, min_df = 0.05, max_features = 500):
    tfidf_vectorizer = TfidfVectorizer(ngram_range = ngram_range,     # unigrams and bigrams (1 word and 2 word units)
                             lowercase =  lowercase,       # why use lowercase?
                             max_df = max_df,           # remove very common words
                             min_df = min_df,           # remove very rare words
                             max_features = max_features)      # keep only top 500 features
    # first we fit the vectorizer to the training data...
    X_train_feats = tfidf_vectorizer.fit_transform(X_train)
    #... then transform our test data
    X_test_feats = tfidf_vectorizer.transform(X_test)
    # get feature names if needed
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    return X_train_feats, X_test_feats, feature_names, tfidf_vectorizer

def bow_vectorizer (X_train, X_test):
    BoW_vectorizer = CountVectorizer()

    X_train_feats_bow = BoW_vectorizer.fit_transform(X_train)

    X_test_feats_bow = BoW_vectorizer.transform(X_test)

    return X_train_feats_bow, X_test_feats_bow, BoW_vectorizer

# Function for saving the vectorizer in the folder models
def save_vectorizer (tfidf_vectorizer, bow_vectorizer):
    from joblib import dump
    dump(tfidf_vectorizer, 'models/tfidf_vectorizer.joblib')
    dump(bow_vectorizer, 'models/bow_vectorizer.joblib')