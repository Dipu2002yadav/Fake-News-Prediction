# Fake News Prediction Project

## Overview

This project aims to build a machine learning model to classify news articles as real or fake. The primary model used in this project is Logistic Regression. The dataset contains news articles with labels indicating whether they are real or fake. The workflow includes data preprocessing, feature extraction, model training, evaluation, and prediction.

## Table of Contents

1. [Dependencies](#dependencies)
2. [Dataset](#dataset)
3. [Data Preprocessing](#data-preprocessing)
4. [Feature Extraction](#feature-extraction)
5. [Model Training](#model-training)
6. [Evaluation](#evaluation)
7. [Prediction System](#prediction-system)

## Dependencies

- numpy
- pandas
- re
- nltk
- sklearn


## Dataset

The dataset used in this project is stored in a CSV file. It contains columns such as `author`, `title`, and `label`. The `label` column indicates whether the news is real (0) or fake (1).

## Data Preprocessing

1. **Loading the Dataset:**
   The dataset is loaded into a Pandas DataFrame.
   
   ```python
   news_dataset = pd.read_csv('/content/drive/MyDrive/train[1].csv')
   ```

2. **Handling Missing Values:**
   Missing values are replaced with empty strings.
   
   ```python
   news_dataset = news_dataset.fillna('')
   ```

3. **Combining Author and Title:**
   A new column `content` is created by merging the `author` and `title`.
   
   ```python
   news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']
   ```

4. **Stemming:**
   The text is cleaned and reduced to its root form using the Porter Stemmer.
   
   ```python
   port_stem = PorterStemmer()
   def stemming(content):
       stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
       stemmed_content = stemmed_content.lower()
       stemmed_content = stemmed_content.split()
       stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
       stemmed_content = ' '.join(stemmed_content)
       return stemmed_content
   news_dataset['content'] = news_dataset['content'].apply(stemming)
   ```

## Feature Extraction

Textual data is converted into numerical data using the TfidfVectorizer.

```python
vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)
```

## Model Training

The Logistic Regression model is trained using the preprocessed data.

```python
model = LogisticRegression()
model.fit(X_train, Y_train)
```

## Evaluation

The accuracy of the model is evaluated on both training and test data.

```python
# Accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

# Accuracy on test data
X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of training data: ', training_data_accuracy)
print('Accuracy score of testing data: ', testing_data_accuracy)
```

## Prediction System

A prediction system is created to classify a given news article as real or fake.

```python
X_new = X_test[202]
prediction = model.predict(X_new)
if prediction == 0:
    print('News is real')
else:
    print('News is fake')
```







