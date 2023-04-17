from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv("train.csv")
print(df)


#count all the rows in file
row_count = sum(1 for rows in df["Id"])
words_Entertainment=''
words_Technology=''
words_Business=''
words_Health=''
stopwords = set(STOPWORDS)

for i in range(row_count):
   if(df["Target"][i]==0):
        words_Entertainment+=str(df["Content"][i])
   elif(df["Target"][i]==1):
        words_Technology+=str(df["Content"][i])
   elif(df["Target"][i]==2):
        words_Business+=str(df["Content"][i])
   elif(df["Target"][i]==3):
        words_Health+=str(df["Content"][i])
        
wordcloud = WordCloud(width = 2000, height = 1000,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10).generate(words_Entertainment)
                
wordcloud1 = WordCloud(width = 2000, height = 1000,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10).generate(words_Technology)
                
wordcloud2 = WordCloud(width = 2000, height = 1000,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10).generate(words_Business)
                
wordcloud3 = WordCloud(width = 2000, height = 1000,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10).generate(words_Health)
                
# plot the WordCloud image                      
plt.figure(figsize = (20, 10), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
 
plt.show()

# plot the WordCloud image                      
plt.figure(figsize = (20, 10), facecolor = None)
plt.imshow(wordcloud1)
plt.axis("off")
plt.tight_layout(pad = 0)
 
plt.show()

# plot the WordCloud image                      
plt.figure(figsize = (20, 10), facecolor = None)
plt.imshow(wordcloud2)
plt.axis("off")
plt.tight_layout(pad = 0)
 
plt.show()

# plot the WordCloud image                      
plt.figure(figsize = (20, 10), facecolor = None)
plt.imshow(wordcloud3)
plt.axis("off")
plt.tight_layout(pad = 0)
 
plt.show()


