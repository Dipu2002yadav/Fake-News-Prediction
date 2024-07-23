# Code by Dipanshu



importing the dependencies
"""

import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import nltk
nltk.download('stopwords')

#printing stopwords in english
print(stopwords.words('english'))

"""data pre_processing"""

#loading the dataset to a pandas dataframe
news_dataset = pd.read_csv('/content/drive/MyDrive/train[1].csv')

news_dataset.shape

#print first 5 rows of data frame

news_dataset.head()

# counting the number of missing values in dataset
news_dataset.isnull().sum()

# replacing null valus with empty string
news_dataset = news_dataset.fillna('')

# merging the author name and new title
news_dataset['content'] = news_dataset['author']+ ' '+news_dataset['title']

print(news_dataset['content'])

# separating the data and label
X = news_dataset.drop('label' , axis=1)
Y = news_dataset['label']

print(X)
print(Y)

"""stemming;

it is process of reducing a word to its root word

example
actor,actress,acting - act
"""

port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

news_dataset['content'] = news_dataset['content'].apply(stemming)

print(news_dataset['content'])

# separating the data and label

X=news_dataset['content'].values
Y=news_dataset['label'].values

print(X)

print(Y)

Y.shape

# converting textual data into numerical data
vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)

print(X)

"""splitting the dataset to training and test data

"""

X_train , X_test ,Y_train , Y_test =train_test_split(X, Y , test_size=0.2 ,stratify =Y, random_state=2)

"""training the model ; logistic regression model"""

model = LogisticRegression()

model.fit(X_train, Y_train)

"""Evaluation

accuracy score
"""

# accuracy score on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction , Y_train)

print('accuracy score of training data: ' ,training_data_accuracy )

# accuracy score on test data
X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction , Y_test)

print('accuracy score of testing data: ' ,testing_data_accuracy )

"""Making a predcting system"""

X_new = X_test[202]

prediction = model.predict(X_new)
print(prediction)

if(prediction ==0):
  print('news is real')
else:
  print('news is fake')

print(Y_test[202])

