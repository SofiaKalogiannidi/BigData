import numpy as np
import re
import nltk
nltk.download('wordnet')
from sklearn.datasets import load_files
nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords
import pandas as pd


 
    
df=pd.read_csv('train.csv')
row_count_train = sum(1 for rows in df["Id"])
y=df["Target"]
y=np.array(y)
#print(y)

documents = []

from nltk.stem import WordNetLemmatizer
stemmer = WordNetLemmatizer()
for i in range(row_count_train):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(df["Content"][i]+df["Title"][i]))
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    # Converting to Lowercase
    document = document.lower()
    # Lemmatization
    document = document.split()
    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    documents.append(document)
    print("TRAIN",i)
    
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=1000, min_df=5, max_df=0.7,binary=True,stop_words=stopwords.words('english'))
#vectorizer = CountVectorizer(stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(documents).toarray()
print(X)

from sklearn.feature_extraction.text import TfidfTransformer
tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()
#print(X)

#training the model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.neighbors import KNeighborsClassifier
#from sklearn.metrics import jaccard
classifier = KNeighborsClassifier(n_neighbors=5,metric='jaccard')

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

#KNN classification
test=pd.read_csv("test.csv")
row_count_test = sum(1 for rows in test["Id"])
docs_test=[]
for i in range(row_count_test):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(test["Content"][i]+test["Title"][i]))
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    # Converting to Lowercase
    document = document.lower()
    # Lemmatization
    document = document.split()
    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    docs_test.append(document)
    print("TEST",i)
    
X_test=vectorizer.transform(docs_test).toarray()
X_test=tfidfconverter.transform(X_test).toarray()
predicted = classifier.predict(X_test)
L=[]
for i in range(len(predicted)):
    tup=[test['Id'][i],predicted[i]]
    print(tup)
    L.append(tup)
classification = pd.DataFrame(L, columns=['Id', 'Predicted'])
classification.to_csv('classification1.csv')