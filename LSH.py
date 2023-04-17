from datasketch import MinHash
from datasketch import MinHashLSHForest, MinHash
import numpy as np
import re
import nltk
nltk.download('wordnet')
from sklearn.datasets import load_files
nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords
import pandas as pd
from datasketch import MinHash, MinHashLSH
from nltk import ngrams
import time
from nltk.stem import WordNetLemmatizer
stemmer = WordNetLemmatizer()

def actual_jaccard(data1,data2):
    s1 = set(data1)
    s2 = set(data2)
    actual_jaccard = float(len(s1.intersection(s2)))/float(len(s1.union(s2)))
    return actual_jaccard
    
def preprocess(data):
    #print(data)
    data= ' '.join(data)
    data = re.sub(r'\W', ' ', data)
    data= re.sub('[^a-zA-Z]', ' ', data)
    data = re.sub(r'\s+[a-zA-Z]\s+', ' ', data)
    data = re.sub(r'\^[a-zA-Z]\s+', ' ', data) 
    data = re.sub(r'\s+', ' ', data, flags=re.I)
    data = re.sub(r'^b\s+', '', data)
    data = data.lower()
    data = data.split()
    data = [stemmer.lemmatize(word) for word in data if word not in stop_words]
   # print(data)
    return data
    
def predict_sentiment(i,result,documents,documents_test,df):
    #similarities=[actual_jaccard(documents[j],documents_test[i]) for j in result]
    #max_value = max(similarities)
    #max_index = similarities.index(max_value)
    
    similarities=[(j,actual_jaccard(documents[j],documents_test[i])) for j in result]
    similarities=Sort_Tuple(similarities)
    similarities=similarities[:15]
    count0=0
    count1=0
    for j in range(len(similarities)):
        if(df['sentiment'][similarities[j][0]]==0):
            count0+=1
        else:
            count1+=1
    if(count0>=count1):
        return 0
    else: 
        return 1
    
    #return df['sentiment'][result[max_index]]
    
def Sort_Tuple(tup): 
  
    tup.sort(key = lambda x: x[1],reverse=True) 
    return tup 
      
    
def KNN_search(i,documents,documents_test):
    distances=[(j,actual_jaccard(documents[j],documents_test[i])) for j in range(len(documents))]
    distances=Sort_Tuple(distances)
    distances=[x[0] for x in distances]
    return distances[:15]
    
stop_words=stopwords.words('english')
df=pd.read_csv('imdb_train.csv')
row_count_train = sum(1 for rows in df["id"])
y=df["sentiment"]
y=np.array(y)
test=pd.read_csv('imdb_test_without_labels.csv')
row_count_test = sum(1 for rows in test["id"])


documents=[]
for i in range(row_count_train):
    data=str(df["review"][i])
    data=data.split()
    data=preprocess(data)
    documents.append(data)
    
start_time = time.time()       
m={}    
for i in range(row_count_train):
    m[i]=MinHash(num_perm=16)
    
for i in range(row_count_train):
    for d in documents[i]:
        m[i].update(d.encode('utf8'))
    
    

lsh = MinHashLSH(threshold=0.2,num_perm=16)

for i in range(len(documents)):
    lsh.insert(i, m[i])

print('It took %s seconds to build structure.' %(time.time()-start_time))

 

documents_test=[]
for i in range(row_count_test):
    data=str(test["review"][i])
    data=data.split()
    data=preprocess(data)
    documents_test.append(data)

  
x={}    
for i in range(row_count_test):
    x[i]=MinHash(num_perm=16)
 
L=[] 
start_time = time.time()
total_right=0
for i in range(row_count_test):
    for d in documents_test[i]:
        x[i].update(d.encode('utf8'))  
  
    result=lsh.query(x[i])
    #print(" Approximate neighbours found:",len(result))
    if(len(result)==0):
        print("no neighbours found")
    right=KNN_search(i,documents,documents_test)
    print("15 Nearest neighbours are:",right)
    common = [c for c in right if c in result]
    print("Neighbors found right are:",len(common))
    total_right+=len(common)
    #print("--------------")
    print(i)
    tup=[test['id'][i],predict_sentiment(i,result,documents,documents_test,df)]
    L.append(tup)
avg_right=total_right/row_count_test
print("average of right predicted neighbours",avg_right)
print('It took %s seconds to query structure.' %(time.time()-start_time))
classification = pd.DataFrame(L, columns=['id', 'sentiment'])
classification.to_csv('LSH.csv')
