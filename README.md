# Sentiment Analysis on IMDB Reviews

**Author**: Vishnupriya Santhosh  
## Overview

This project focuses on performing sentiment analysis on the IMDB dataset using various machine learning and deep learning models. The goal is to classify movie reviews as either positive or negative.

## Dataset

The dataset contains 50,000 rows with two attributes: `review` and `sentiment`. Each review is labeled as either positive or negative.

## Installation

To run this project, you need to have Python installed along with the following libraries:

```bash
pip install pandas nltk tensorflow scikit-learn matplotlib seaborn

Preprocessing Steps
Tokenization: Breaking down text into individual tokens.
Removal of Stop Words: Eliminating common words that do not contribute to sentiment.
Lemmatization: Converting words to their base forms.
Removal of Special Characters: Cleaning the text by removing unnecessary characters.
Data Representation
Bag of Words (BoW): Using CountVectorizer to create a bag-of-words representation.
TF-IDF: Using TfidfVectorizer to create a TF-IDF representation.
Models Used
Logistic Regression:
Achieved accuracy:
BoW: ...
TF-IDF: ...
Support Vector Machine (SVM):
Achieved accuracy:
BoW: ...
TF-IDF: ...
Multinomial Naive Bayes (MNB):
Achieved accuracy:
BoW: ...
TF-IDF: ...
Convolutional Neural Network (CNN):
Achieved accuracy: ...
Results
The models' performances were compared using accuracy scores and classification reports. The CNN model achieved the highest accuracy of 85%.

Visualization
A bar chart comparing the test accuracies of different classifiers is provided.

Conclusion
From the analysis, the CNN model outperformed other models with the highest accuracy. Both Bag of Words and TF-IDF are effective features, with a slight edge for BoW in some models.

How to Run
Clone the repository.
Install the required libraries.
Run the Jupyter notebook or Python script to see the preprocessing, model training, and evaluation steps.
