import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer # to convert textual data to numerical data using tfidf vectorizer function 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import MultinomialNB 
from sklearn import metrics # to calculate statistics of the classification 

# read the training file and store text and id entires in their own lists 
data=pd.read_csv('train3.csv')
textlist=list(data.Text)
idlist=list(data.ID)

# use count vertorizer 
count=CountVectorizer()
word_count=count.fit_transform(textlist) 
print("___________________________________IDF")
print(word_count.shape) # will reveal how many sentences and unique words 

# use idf transformer to get the weights of all unique words in the documents 
tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count)
df_idf = pd.DataFrame(tfidf_transformer.idf_,index=count.get_feature_names_out(),columns=['idf_weights'])
# inverse document frequency 
df_idf.sort_values(by=['idf_weights'])
print("___________________________________DF_IDF")
print(df_idf)

# tf-idf transformation 
tf_idf_vector = tfidf_transformer.transform(word_count)
feature_names = count.get_feature_names_out()
first_document_vector = tf_idf_vector[1]
df_tfidf = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])
# inverted document frequency (IDF)
df_tfidf.sort_values(by=["tfidf"],ascending=False)
print(df_tfidf)