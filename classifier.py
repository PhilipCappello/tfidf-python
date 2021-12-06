import numpy as np 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import f1_score
import csv

# reading the data
test_csv = pd.read_csv('test3.csv') 
train_csv = pd.read_csv('train3.csv')

#stopword removal and lemmatization
stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

nltk.download('stopwords')

train_csv.head()

train_X = train_csv['Text']
X = []
for i in range(0, len(train_X)):
    review = re.sub('[^a-zA-Z]', ' ', train_X[i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords)]
    review = ' '.join(review)
    X.append(review)

y = train_csv['Class']

# Building a TF IDF matrix out of the corpus of reviews
from sklearn.feature_extraction.text import TfidfVectorizer
td = TfidfVectorizer(max_features = 5000)
X = td.fit_transform(X).toarray()

# Splitting into training & test subsets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
                                                    random_state = 0)

# Training the classifier & predicting on test data
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test) # Predicted Result 
# Classification metrics
from sklearn.metrics import accuracy_score, classification_report
classification_report = classification_report(y_test, y_pred)

print('\n Accuracy: ', accuracy_score(y_test, y_pred))
print('\nClassification Report')
print('======================================================')
print('\n', classification_report)

# testing with actual test csv

X = test_csv['Text']
test_X = []
for i in range(0, len(X)):
    review = re.sub('[^a-zA-Z]', ' ', X[i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords)]
    review = ' '.join(review)
    test_X.append(review)
X_ID = test_csv['ID']

test_X = td.fit_transform(test_X).toarray()

y_pred_test_data = classifier.predict(test_X)

f = open('prediction.csv', 'w')
writer = csv.writer(f)
for i in range(0, len(y_pred_test_data)):
    row = [X_ID[i], y_pred_test_data[i]]
    writer.writerow(row)
f.close()
