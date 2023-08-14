import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Function for importing and splitting data

def import_data():
    """Import data from a csv file and split it into train and test sets."""
    filename = os.path.join('in', 'fake_or_real_news.csv')
    data = pd.read_csv(filename, index_col=0)
    X = data['text']
    y = data['label']
    return X, y

# Function for splitting data into test/train
def train_test_split_func (X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size=test_size,
                                                        random_state=42) # random_state is the seed used by the random number generator
    return X_train, X_test, y_train, y_test

# Class for printing colored text (not directly related to preprocessing)
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'