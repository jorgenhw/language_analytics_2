<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h1 align="center">Cultural Datascience 2023</h1> 
  <h2 align="center">Assignment 2</h2> 
  <h3 align="center">Language Analytics</h3> 


  <p align="center">
    Jørgen Højlund Wibe
  </p>
</p>


<!-- ABOUT THE PROJECT -->
## About the project
This assignment compares two different methods to classify whether a news article (or snippet) is fake or real. The two methods are logistic regression and neural networks.
This is done by firstly vectorizing the data with a TF-IDF vectorizer (BoW is available) and then training both the neural network and the logistic regression classifier on the labelled dataset.

<!-- USAGE -->
## Usage

To use or reproduce our results you need to adopt the following steps.

**NOTE:** There may be slight variations depending on the terminal and operating system you use. The following example is designed to work using the Visual Studio Code version 1.76.0 (Universal). The terminal code should therefore work using a unix-based bash. The avoid potential package conflicts, the ```setup.sh``` bash files contains the steps necesarry to create a virtual environment for the project.

1. Clone repository
2. Run setup.sh
3. [OPTIONAL] Change vectorizer method

### Clone repository

Clone repository using the following lines in the unix-based bash:

```bash
git clone https://github.com/jorgenhw/language_analytics_assignment_2.git
cd language_analytics_assignment_2
```

### Run ```setup.sh```

To replicate the results, I have included a bash script that automatically 

1. Creates a virtual environment for the project
2. Activates the virtual environment
3. Installs the correct versions of the packages required
4. Runs the script
5. Deactivates the virtual environment

Run the code below in your bash terminal:

```bash
bash run.sh
```

### [OPTIONAL] Change vectorizer method to BoW
Per default, the script is set to use TF-IDF vectorized data. To use Bag of Words instead, you will need to uncomment line 67 and 83 in ```main.py``` as well as changing the input of the function as specified in the comment on line 71 and 87, also in ```main.py```.

### Changing arguments via ```argparse```
To provide more flexibility and enable the user to change the parameters of the script from the command line, we have implemented argparse in our script. This means that by running the script with specific command line arguments, you can modify parameters such as the batch size, the number of epochs to train the model, and the learning rate.

To see all the available arguments, simply run the command:

```bash
python main.py --help
```
This will display a list of all the available arguments and their default values.


## Inspecting results

A classification report from the each approach on the data is located in the folder ```out```. Here one can inspect the results.

## Using the classifiers to predict new sentences

If one wishes to apply these trained models on new data, one can find the trained models in the folder ```models```. To apply on new data, follow the below step in a python kernel: 
```
from joblib import dump, load
loaded_log_reg_model = load("models/log_reg.joblib")
loaded_nn_model = load("models/nn.joblib")
loaded_vect = load("models/tfidf_vectorizer.joblib")

sentence = "Hilary Clinton is a crook who eats babies!"

test_sentence = loaded_vect.transform([sentence])
loaded_log_reg_model.predict(test_sentence)
```

<!-- REPOSITORY STRUCTURE -->
## Repository structure
This repository has the following structure:
```
│   main.py
│   README.md
│   requirements.txt
│   run.sh
│
├───in
│       fake_or_real_news.csv
│
├───models
│       bow_vectorizer.joblib
│       log_reg.joblib
│       nn.joblib
│       tfidf_vectorizer.joblib
│
├───out
│       classification_report_log_reg.csv
│       classification_report_nn.csv
│
└──src
        log_reg_classification.py
        nn_classification.py
        vectorizers.py

```


<!-- DATA -->

## Remarks on findings
This study found that neural networks only slightly outperformed logistic regression at classifying fake vs real data - when data is vectorized using TF-IDF. 

This suggests that for this particular classification task, the additional complexity and computational resources required to train a neural network may not be worth the modest improvement in accuracy over logistic regression.