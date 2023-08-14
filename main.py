"""
Main script for running the fake news detection pipeline.
"""

####### PACKAGES #######

# Importing required packages
import argparse

# Importing functions from folder src
import src.log_reg_classification as log_reg
import src.nn_classification as nn
import src.vectorizers as vec
import src.preprocessing as pre


####### MAIN FUNCTION #######
def main(args):
    """
    PREPROCESSING DATA
    """
    # Import data
    print(pre.bcolors.HEADER + "[STATUS] Importing data..." + pre.bcolors.ENDC)
    X, y = pre.import_data()
    print(pre.bcolors.OKGREEN + "[STATUS] Successfully imported data" + pre.bcolors.ENDC)

    # Splitting data
    print(pre.bcolors.HEADER + "[STATUS] Splitting data..." + pre.bcolors.ENDC)
    X_train, X_test, y_train, y_test = pre.train_test_split_func(X, y, 
                                                                 test_size=args.test_size)
    print(pre.bcolors.OKGREEN + "[STATUS] Successfully split data" + pre.bcolors.ENDC)
    
    """
    VECTORIZING DATA
    """
    # Vectorizing data using TF-IDF
    print(pre.bcolors.HEADER + "[STATUS] Vectorizing data using TF-IDF..." + pre.bcolors.ENDC)
    X_train_feats, X_test_feats, feature_names, tfidf_vectorizer = vec.tfidf_vectorize_data(
        X_train, X_test, 
        ngram_range=args.ngram_range, 
        lowercase=args.lowercase, 
        max_df=args.max_df, 
        min_df=args.min_df, 
        max_features=args.max_features
    )
    print(pre.bcolors.OKGREEN + "[STATUS] Successfully vectorized data using TF-IDF" + pre.bcolors.ENDC)
    
    # Vectorizing data using bag of words (BoW) - only for optional comparison
    print(pre.bcolors.HEADER + "[STATUS] Vectorizing data using Bag of Words (BoW)..." + pre.bcolors.ENDC)
    X_train_feats_bow, X_test_feats_bow, bow_vectorizer = vec.bow_vectorizer(X_train, X_test)
    print(pre.bcolors.OKGREEN + "[STATUS] Sucess! Data vectorized using BoW" + pre.bcolors.ENDC)

    # Save vectorizer
    print(pre.bcolors.HEADER + "[STATUS] Saving vectorizer..." + pre.bcolors.ENDC)
    vec.save_vectorizer(tfidf_vectorizer, bow_vectorizer)
    print(pre.bcolors.OKGREEN + "[STATUS] Sucess! Vectorizer saved" + pre.bcolors.ENDC)

    """
    TRAINING AND EVALUATING MODEL USING TF-IDF VECTORIZER AND LOGISTIC REGRESSION
    """
    # Train and evaluate logistic regression model
    print(pre.bcolors.HEADER + "[STATUS] Training and evaluating logistic regression model..." + pre.bcolors.ENDC)
    y_pred = log_reg.classification_log_reg(X_train_feats, y_train, X_test_feats)
    print(pre.bcolors.OKGREEN + "[STATUS] Sucess! Logistic regression model trained and evaluated" + pre.bcolors.ENDC)

    # [OPTIONAL]: Uncomment below line to train the model and getting predictions using bag of words vectorizer
    # y_pred_bow = log_reg.classification_log_reg (X_train_feats_bow, y_train, X_test_feats_bow)
    
    # Save classification report from logistic regression
    print(pre.bcolors.HEADER + "[STATUS] Saving classification report from logistic regression..." + pre.bcolors.ENDC)
    log_reg.classification_report(y_test, y_pred) # Here using tfidf vectorizer. To use bow, change y_pred to y_pred_bow
    print(pre.bcolors.OKGREEN + "[STATUS] Sucess! Classification report saved" + pre.bcolors.ENDC)

    """
    TRAINING AND EVALUATING MODEL USING TF-IDF VECTORIZER AND NEURAL NETWORK
    """
    # Train and evaluate neural network model
    print(pre.bcolors.HEADER + "[STATUS] Training and evaluating neural network model..." + pre.bcolors.ENDC)
    y_pred = nn.train_model(X_train_feats, y_train, X_test_feats)
    print(pre.bcolors.OKGREEN + "[STATUS] Sucess! Neural network model trained and evaluated" + pre.bcolors.ENDC)
    
    # [OPTIONAL]: Uncomment below line to train the model and getting predictions using bag of words vectorizer
    # y_pred_bow = log_reg.classification_rog_reg (X_train_feats_bow, y_train, X_test_feats_bow)
    
    # Save classification report from neural network
    print(pre.bcolors.HEADER + "[STATUS] Saving classification report from neural network..." + pre.bcolors.ENDC)
    nn.classification_report(y_test, y_pred) # Here using tfidf vectorizer. To use bow, change y_pred to y_pred_bow
    print(pre.bcolors.OKGREEN + "[STATUS] Sucess! Script has run succesfully and files has been saved" + pre.bcolors.ENDC)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for training and evaluating a machine learning model for fake news detection.')
    parser.add_argument('--test_size', type=float, default=0.2, help='specify the size of the test/train split, default is 0.2')
    parser.add_argument('--ngram_range', type=tuple, nargs=2, default=(1,2), help='ngram range for the vectorizer, default is (1,2): unigrams and bigrams (1 word and 2 word units)')
    parser.add_argument('--lowercase', type=bool, default=True, help='whether to convert all text to lowercase, default is True')
    parser.add_argument('--max_df', type=float, default=0.95, help='maximum document frequency for the vectorizer, default is 0.95')
    parser.add_argument('--min_df', type=float, default=0.05, help='minimum document frequency for the vectorizer, default is 0.05')
    parser.add_argument('--max_features', type=int, default=500, help='maximum number of features for the vectorizer, default is 500')
    args = parser.parse_args()
    main(args)