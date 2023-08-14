"""
### IMPORTS ###
"""
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import os
import pandas as pd

"""
#### FUNCTIONS ####
"""
# Function for training the model and getting predictions
def train_model(X_train_feats, y_train, X_test_feats):
    """
    : X_train_feats: numpy array of shape (n_train, d)
    : y_train: numpy array of shape (n_train,)
    : X_test_feats: numpy array of shape (n_test, d)
    """
    classifier = MLPClassifier(activation = "relu", # logistic -> relu  
                           hidden_layer_sizes = (100,20), # 100 refers to the number of neurons in the first layer, 20 refers to the number of neurons in the second layer
                           solver='adam', # added
                           learning_rate='adaptive', # added
                           max_iter=1000, # increase the number of iterations
                           random_state = 42
                           )
    # Fit the model to the training data
    classifier.fit(X_train_feats, y_train)
    # Saving the model to disk
    from joblib import dump
    dump(classifier, 'models/nn.joblib')
    # get predictions from the test data
    y_pred = classifier.predict(X_test_feats)
    return y_pred

# Function for evaluating the model on the test set, printing the classification report and saving it to a csv file in out
def classification_report (y_test, y_pred):
    """
    : y_test: numpy array of shape (n_test,)
    : y_pred: numpy array of shape (n_test,)
    """
    classifier_metrics = metrics.classification_report(y_test, y_pred, output_dict=True)
    df = pd.DataFrame(classifier_metrics).transpose()
    # Round the values in df to 2 decimals
    df = df.round(2)
    df.to_csv(os.path.join('out', 'classification_report_nn.csv'), index=True)