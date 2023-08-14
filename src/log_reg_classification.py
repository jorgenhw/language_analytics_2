"""
IMPORTS
"""
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import os
import pandas as pd

"""
#### FUNCTIONS ####
"""
# Function for training the model and getting predictions
def classification_log_reg (X_train_feats, y_train, X_test_feats):
    """
    Function for training the model and getting predictions
    """
    # Calling classifier
    classifier = LogisticRegression(random_state=42).fit(X_train_feats, y_train)
    # get predictions from the test data
    y_pred = classifier.predict(X_test_feats)
    # save the model to disk
    from joblib import dump
    dump(classifier, 'models/log_reg.joblib')
    
    return y_pred

# Getting prediction metrics
def classification_report(y_test, y_pred):
    """
    : y_test: numpy array 
    : y_pred: numpy array 
    """
    classifier_metrics = metrics.classification_report(y_test, y_pred, output_dict=True)
    df = pd.DataFrame(classifier_metrics).transpose()
    # Round the values in df to 2 decimals
    df = df.round(2)
    df.to_csv(os.path.join('out', 'classification_report_log_reg.csv'), index=True)
    #print(classifier_metrics)