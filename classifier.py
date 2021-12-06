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


#reading the data
test_csv = pd.read_csv('small_test3.csv') 
train_csv = pd.read_csv('small_train3.csv')

#stopword removal and lemmatization
stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

nltk.download('stopwords')

train_csv.head()

train_X_non = train_csv['Text']   # '0' refers to the review text
train_y = train_csv['ID']   # '1' corresponds to Label (1 - positive and 0 - negative)
test_X_non = test_csv['Text']
test_y = test_csv['ID']
train_X=[]
test_X=[]

# text pre processing
# Removes numbers and symbols that are not letters 
# Lower cases all text
# Splits up all words inividually and removes stopwords
for i in range(0, len(train_X_non)):
    review = re.sub('[^a-zA-Z]', ' ', train_X_non[i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords)]
    review = ' '.join(review)
    train_X.append(review)

print("Line:46")

# text pre processing
# Same shit different language
for i in range(0, len(test_X_non)):
    review = re.sub('[^a-zA-Z]', ' ', test_X_non[i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords)]
    review = ' '.join(review)
    test_X.append(review)

print("Line:58")
# print(train_X[10]) -> OUTPUT: mmmmmm bobbie thanksgiving sandwich capistrami oh boy place delicious

# Now, we use the TF-IDF Vectorizer.
#tf idf
tf_idf = TfidfVectorizer()
#applying tf idf to training data
X_train_tf = tf_idf.fit_transform(train_X)
#applying tf idf to training data
X_train_tf = tf_idf.transform(train_X)

print("Line:70")

# print("n_samples: %d, n_features: %d" % X_train_tf.shape) -> OUTPUT: n_samples: 56000, n_features: 47595

# Now, we transform the test data into TF-IDF matrix format.
X_test_tf = tf_idf.transform(test_X)

# print("n_samples: %d, n_features: %d" % X_test_tf.shape) -> OUTPUT: n_samples: 14000, n_features: 47595

# Now we can proceed with creating the classifier.
# We shall be creating a Multinomial Naive Bayes model. This algorithm is based on Bayes Theorem.

#naive bayes classifier
print("Line:82")
naive_bayes_classifier = MultinomialNB()
print("Line:84")
naive_bayes_classifier.fit(X_train_tf, train_y)

print("Line:87")
#predicted y
y_pred = naive_bayes_classifier.predict(X_test_tf)

print(metrics.classification_report(test_y, y_pred, target_names=['positive', 'negative','neutral']))