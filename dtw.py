import pandas as pd
import numpy as np
import math
import numpy as np
import time

        
def DTWdistance(seriesa,seriesb):
 
    n=len(seriesa)
    m=len(seriesb)
    DTW = np.zeros((n+1, m+1))
    for i in range(n+1):
        for j in range(m+1):
            DTW[i,j]=np.inf
    DTW[0,0]=0
    
    for i in range(1,n+1):
        for j in range(1,m+1):
            dist=abs(seriesa[i-1]-seriesb[j-1])
            DTW[i,j]=dist+ np.min([DTW[i-1,j],DTW[i,j-1],DTW[i-1,j-1]])


    return DTW[n,m]
    
    
L=[]
df=pd.read_csv("dtw_test.csv")
row_count= sum(1 for rows in df["id"])
start_time = time.time()  
for i in range(row_count):
  
    reviewa= df["series_a"][i]
    reviewa=reviewa.replace('[', '')
    reviewa=reviewa.replace(',', '')
    reviewa=reviewa.replace(']', '')
    reviewa=reviewa.split()
    reviewa=[float(x) for x in reviewa]
   
    reviewb= df["series_b"][i]
    reviewb=reviewb.replace('[', '')
    reviewb=reviewb.replace(',', '')
    reviewb=reviewb.replace(']', '')
    reviewb=reviewb.split()
    reviewb=[eval(x) for x in reviewb]
   
    tup=[df["id"][i],DTWdistance(reviewa,reviewb)]
    print(tup)
    L.append(tup)
print('It took %s seconds for all test series.' %(time.time()-start_time))
result = pd.DataFrame(L, columns=['id', 'distance'])
result.to_csv('dtw.csv')
