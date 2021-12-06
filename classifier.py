#importing libraries
from nltk import text
import numpy as np 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import nltk
import re
import string
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

X = train_csv['Text']
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
X_ID = test_csv['ID']

X = td.fit_transform(X).toarray()

y_pred_test_data = classifier.predict(X)

f = open('prediction.csv', 'w')
writer = csv.writer(f)
for i in range(0, len(y_pred_test_data)):
    row = [X_ID[i], y_pred_test_data[i]]
    writer.writerow(row)
f.close()


# train_X_non = train_csv['Text']   # '0' refers to the review text
# train_y = train_csv['Class']   # '1' corresponds to Label (1 - positive and 0 - negative)
# test_X_non = test_csv['Text']
# test_y = test_csv['Class']
# train_X=[]
# test_X=[]

# # text pre processing
# # Removes numbers and symbols that are not letters 
# # Lower cases all text
# # Splits up all words inividually and removes stopwords
# for i in range(0, len(train_X_non)):
#     review = re.sub('[^a-zA-Z]', ' ', train_X_non[i])
#     review = review.lower()
#     review = review.split()
#     review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords)]
#     review = ' '.join(review)
#     train_X.append(review)

# print("Line:46")

# # text pre processing
# # Same shit different language
# for i in range(0, len(test_X_non)):
#     review = re.sub('[^a-zA-Z]', ' ', test_X_non[i])
#     review = review.lower()
#     review = review.split()
#     review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords)]
#     review = ' '.join(review)
#     test_X.append(review)

# print("Line:58")
# # print(train_X[10]) -> OUTPUT: mmmmmm bobbie thanksgiving sandwich capistrami oh boy place delicious

# # Now, we use the TF-IDF Vectorizer.
# #tf idf
# tf_idf = TfidfVectorizer()
# #applying tf idf to training data
# X_train_tf = tf_idf.fit_transform(train_X)
# #applying tf idf to training data
# X_train_tf = tf_idf.transform(train_X)

# print("Line:70")

# # print("n_samples: %d, n_features: %d" % X_train_tf.shape) -> OUTPUT: n_samples: 56000, n_features: 47595

# # Now, we transform the test data into TF-IDF matrix format.
# X_test_tf = tf_idf.transform(test_X)

# # print("n_samples: %d, n_features: %d" % X_test_tf.shape) -> OUTPUT: n_samples: 14000, n_features: 47595

# # Now we can proceed with creating the classifier.
# # We shall be creating a Multinomial Naive Bayes model. This algorithm is based on Bayes Theorem.

# #naive bayes classifier
# print("Line:82")
# naive_bayes_classifier = MultinomialNB()
# print("Line:84")
# naive_bayes_classifier.fit(X_train_tf, train_y)

# print("Line:87")
# #predicted y
# y_pred = naive_bayes_classifier.predict(X_test_tf)

# # y_train = naive_bayes_classifier.predict(X_train_tf)


# # Classification metrics

# #classification_report = classification_report(test_y, y_pred)
# # target_names = [f"class{i}" for i in range(5)]

# #print(metrics.classification_report(test_y, y_pred, target_names=['positive','negative','neutral']))
# print(metrics.f1_score(test_y, y_pred, average='weighted', labels=np.unique(y_pred)))
