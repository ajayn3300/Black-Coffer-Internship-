#!/usr/bin/env python
# coding: utf-8

# #  Objective
#     
#    **~ The objective of this assignment is to extract textual data articles from the given URL and perform text analysis to compute variables that are explained below.**
# 

# In[919]:


import pandas as pd
import requests
from bs4 import BeautifulSoup
import glob
from gensim.utils import simple_preprocess
from collections import Counter

import nltk
from nltk.corpus import stopwords

import pyphen
import re


# #  Data Extraction 
#    **~ Input.xlsx \
# For each of the articles, given in the input.xlsx file, extract the article text and save the extracted article in a text file with URL_ID as its file name.
# While extracting text, please make sure your program extracts only the artic
# le title and the article text. It should not extract the website header, footer, or anything other than the article text.**

# In[524]:


# read input
links=pd.read_excel(r'C:\Users\ajay\Desktop\Blackcoffer\Input.xlsx')


# In[525]:


# data head
links.head()


# In[526]:


# creating a header for accesing request
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:55.0) Gecko/20100101 Firefox/55.0',
}


# **Extracting Articles**

# In[527]:


#saving articles as txt files 
for i in links['URL_ID'].values:
    
    # creating path for writing txt file
    f=open('E:\parsed_article\{}.txt'.format(int(i)),'w')
    g=open('E:\parsed_article\sub-heads\{}.txt'.format(int(i)),'w')
    
    # getting page request and content 
    req=requests.get(links[links['URL_ID']==i][['URL']].values[0][0],headers=headers)
    page=BeautifulSoup(req.content,'lxml')
    
    # applying a condition
    if page.find('div',class_="td-404-title")==None:
        
        f.write(page.find('h1').text+'\n')
        for j in page.find_all('p'):
            f.write(str(j.text.encode('utf-8'))+'\n')     # here i am writing encoded text to the fileso it don't throw Unicode error
        
        for l in page.find_all('strong'):                 # extracting sub-heads so we can remove these from text to get accuracte no. of sentences
            g.write(str(l.text.encode('utf-8'))+'\n')
    else: 
        
        f.write('Ooops... Error 404')
        g.write('Ooops... Error 404')
    
    f.flush()
    g.flush()
    f.close()
    g.close()


#       Above i have applied a condition before writing the text file because if there are any pages which are empty or which are not found  then it will write 'Ooops....Error 404' to the text file and if page is available then it will write article included to the text file.  

# # 1.	Sentimental Analysis
#    *~ Sentimental analysis is the process of determining whether a piece of writing is positive, negative, or neutral. The below Algorithm is designed for use in Financial Texts. It consists of steps:*
# 

# **1.1** 	**Cleaning using Stop Words Lists**\
#         *The Stop Words Lists (found in the folder StopWords) are used to clean the text so that Sentiment Analysis can be performed by excluding the words found in Stop Words List.*

# In[528]:


# let's create a dictionary which will contain URL_id as key and article as text
article=dict()

for i in links['URL_ID'].values:
    article[i]=open(r'E:\parsed_article\{}.txt'.format(int(i)),'r').read().splitlines()
    


# In[529]:


# removing sub-heads from articles
for i in links['URL_ID'].values:
    article[int(i)]=[l for l in article[int(i)] if l not in open(r'E:\parsed_article\sub-heads\{}.txt'.format(int(i)),'r').read().splitlines()]


# In[534]:





# In[535]:


# adding the article columns to links dataframe so it could be easy for text analysis 
links=links.join(pd.Series(list(article.values()),index=links.index,name='article'))


# In[536]:


# let's check
links.head(10)


# In[559]:


# check how many pages are empty 
links[links['article'].str.len()==0]


# In[560]:


# let's remove them
links=links[links['article'].str.len()!=0]


# In[573]:


# lets make simple preprocess function   NoTE: it will not remove stopwords
def pre_process(x):
    p=[]
    for i in x:
        p.append(' '.join(simple_preprocess(i)))
    
    return p
    


# In[577]:


# apply pre_process
links['article']=links['article'].apply(lambda x:pre_process(x))


# In[579]:


# let's save stopwords path
stopwords_path=glob.glob(r'C:\Users\ajay\Desktop\Blackcoffer\StopWords\*.txt')


# In[580]:


# saving all stopwords in one variable 
stopwords=''
for i in stopwords_path:

    stopwords+=(' '.join(open(i,'r').read().lower().split()))
    
stopwords=' '.join([i for i in stopwords.split() if i!='|'])


# In[584]:


# let's make a function for removing stopwords
def remo_stopwords(x):
    p=[]
    for i in x:
        l=[]
        for j in i.split():
            if j not in stopwords.split():
                l.append(j)
        p.append(' '.join(l))
    
    return p
    


# In[587]:


# apply stopwords remover
links['article']=links['article'].apply(lambda x:remo_stopwords(x))


# In[592]:


len(links['article'][0])


# **1.2** **Creating a dictionary of Positive and Negative words**\
#    *~ The Master Dictionary (found in the folder MasterDictionary) is used for creating a dictionary of Positive and Negative words. We add only those words in the dictionary if they are not found in the Stop Words Lists.*

# In[594]:


# let's read first negative and positive words
negative_words=open(r'C:/Users/ajay/Desktop/Blackcoffer/MasterDictionary/negative-words.txt','r').read().split()
positive_words=open(r'C:/Users/ajay/Desktop/Blackcoffer/MasterDictionary/positive-words.txt','r').read().split()


# In[596]:


#sample
negative_words[:5]


# In[597]:


#let's make a function which will count positive and negative words
def count_pos_neg(x):
    pos=[]
    neg=[]
    for i in x:
        for j in i.split():
            if j in negative_words:
                neg.append(j)
            if j in positive_words:
                pos.append(j)
    return {'pos':pos,
           'neg':neg}


# In[601]:


# let's add a columns to dataframe which will include + and - words 
links['pos_neg']=links.article.apply(lambda x:count_pos_neg(x))


# In[605]:


#sample
links.head(3)


# **1.3	Extracting Derived variables**\
#     We convert the text into a list of tokens using the nltk tokenize module and use these tokens to calculate the 4 variables described below:\
#     \
# ***Positive Score:*** This score is calculated by assigning the value of +1 for each word if found in the Positive Dictionary and then adding up all the values.\
# \
# ***Negative Score:*** This score is calculated by assigning the value of -1 for each word if found in the Negative Dictionary and then adding up all the values. We multiply the score with -1 so that the score is a positive number.\
# \
# ***Polarity Score:*** This is the score that determines if a given text is positive or negative in nature. It is calculated by using the formula:\ 
# Polarity Score = (Positive Score – Negative Score)/ ((Positive Score + Negative Score) + 0.000001)\
# Range is from -1 to +1\
# \
# ***Subjectivity Score:*** This is the score that determines if a given text is objective or subjective. It is calculated by using the formula:\
# Subjectivity Score = (Positive Score + Negative Score)/ ((Total Words after cleaning) + 0.000001)\
# Range is from 0 to +1
# 
# 

# In[606]:


# adding variable called Postitve Score

links['Positive Score']=links['pos_neg'].apply(lambda x:len(x['pos']))


# In[609]:


#adding variable called Negative Score
links['Negative Score']=links['pos_neg'].apply(lambda x:len(x['neg']))


# In[622]:


# calculating Polarity 

# define func
def polarity_score(df):
    score=[]
    for i in df.index:
        score.append((df['Positive Score'][i]-df['Negative Score'][i])/(df['Positive Score'][i]+df['Negative Score'][i])+0.000001)
    return score

# adding Polarity Score to dataframe

links['Polarity Score']=polarity_score(links)


# In[634]:


# calculating Subjectivity Score

# define a function

def subject_calc(df):
    # let's find out words counts 
    coun=[]
    for i in df.index:
        l=0
        for j in df['article'][i]:
            l+=len(j.split())
        coun.append(l)
    coun=pd.Series(coun,index=df.index)
    
    # let's find out score now
    score=[]
    for i in df.index:
        score.append((df['Positive Score'][i]+df['Negative Score'][i])/coun[i]++ 0.000001)
    
    return score


#let's add feature calleld Subjectivity Score
links['Subjectivity Score']=subject_calc(links)


# In[635]:


links


# # 2.	Analysis of Readability
#   *Analysis of Readability is calculated using the Gunning Fox index formula described below.*
#   

# **2.1 Average Sentence Length**\
#    the number of words / the number of sentences

# In[713]:


# removing stopwords if any remaining 
links['article']=links['article'].apply(lambda x:[i for i in ' '.join(x).split() if i not in stopwords.words('english')])


# In[761]:


# calculating average sentence length
Average_Sentence_Length=[]

for i in links.URL_ID.values:
    Average_Sentence_Length.append(round(len(links[links['URL_ID']==int(i)].article.values[0])/len(nltk.sent_tokenize(open(r'E:\parsed_article\{}.txt'.format(int(i)),'r').read()))))
     


# In[762]:


# Let's create feature of that
links['Average_Sentence_Length']=Average_Sentence_Length


# **2.2 Percentage of Complex words**\
#    the number of complex words / the number of words 

# In[827]:


# let;s make a function which will calculate % of complex words

def calc_complex_percent(x):
    
    # passing all the words to class
    wor=[pyphen.Pyphen(lang='en').inserted(i) for i in x]

    #getting only syllable words
    phens=[i for i in wor if '-' in i]
    
    return len(phens)/len(x)

        


# In[828]:


# let's add a feature called PERCENTAGE OF COMPLEX WORDS
links['PERCENTAGE OF COMPLEX WORDS']=links.article.apply(lambda x :calc_complex_percent(x))


# **2.3 Fog Index**\
# = 0.4 * (Average Sentence Length + Percentage of Complex words)

# In[844]:


# let's define a function for calculating Fog Index
def fog_ind(df):
    fog=[]
    for i in df.index:
        fog.append(0.4*(df[df.index==i]['Average_Sentence_Length'][i]+df[df.index==i]['PERCENTAGE OF COMPLEX WORDS'][i]))
    
    return fog


# In[845]:


# let's create a feature as Fog Index

links['Fog Index']=fog_ind(links)


# In[849]:


links.head(3)


# # 3	Average Number of Words Per Sentence
# The formula for calculating is:\
# *Average Number of Words Per Sentence = the total number of words / the total number of sentences*

# In[866]:


# let's calculate
avg=[]
for i in links['URL_ID'].values:
    avg.append(round(len(links[links['URL_ID']==i]['article'].values[0])/len(nltk.sent_tokenize(open(r'E:\parsed_article\{}.txt'.format(int(i)),'r').read()))))

    
# add feature AVG NUMBER OF WORDS PER SENTENCE

links['AVG NUMBER OF WORDS PER SENTENCE']=avg


# In[1065]:


links.head(2)


# # 4	Complex Word Count
# Complex words are words in the text that contain more than two syllables.
# 

# In[868]:


# let;s make a function which will count complex words

def count_complex(x):
    
    # passing all the words to class
    wor=[pyphen.Pyphen(lang='en').inserted(i) for i in x]

    #getting only syllable words
    phens=[i for i in wor if '-' in i]
    
    return len(phens)   


# In[869]:


# let's add feature COMPLEX WORD COUNT
links['COMPLEX WORD COUNT']=links.article.apply(lambda x:count_complex(x))


# In[871]:


#sample
links.head(2)


# # 5	Word Count
# We count the total cleaned words present in the text by 
# 1.	removing the stop words (using stopwords class of nltk package).
# 2.	removing any punctuations like ? ! , . from the word before counting.
# 

# In[877]:


# we have already cleaned our data,so we will directly count words

links['WORD COUNT']=links.article.apply(lambda x:len(x))


# # 6.SYLLABLE PER WORD
# We count the number of Syllables in each word of the text by counting the vowels present in each word. We also handle some exceptions like words ending with "es","ed" by not counting them as a syllable.

# In[914]:


# we will create a dictionary with keys as words and syllable count as values

# create  function

def syll_per_word(x):
    #creating an empty dic
    dic={}    
    
    #selecting unique words only 
    unique=list(set(x))    
    
    # exception 
    exception=[i for i in unique if i[-2:] not in ['es','ed']]
    
    # complex words
    syll=[i for i in [pyphen.Pyphen(lang='en').inserted(i) for i in exception ] if '-' in i]
    
    # counting syllables per word
    for i in syll:
        dic[i.replace('-','')]=len(i.split('-'))
    
    return dic


# In[916]:


# let's add feature to pur dataframe

links['SYLLABLE PER WORD']=links.article.apply(lambda x:syll_per_word(x))


# In[918]:


#sample
links.head(3)


# # 7.	Personal Pronouns
# To calculate Personal Pronouns mentioned in the text, we use regex to find the counts of the words - “I,” “we,” “my,” “ours,” and “us”. Special care is taken so that the country name US is not included in the list.
# 

# In[1059]:


# let's count
PERSONAL_PRONOUNS=[]    # here we will append pronouns count

# let's create a compiler first,
pronouns=re.compile(r'\bi\b|\bwe\b|\bmy\b|\bours\b|\bus\b',re.IGNORECASE)

# count and save
for i in links['URL_ID'].values:
    l=pronouns.findall(open(r'E:\parsed_article\{}.txt'.format(int(i)),'r').read())
    s=len([i for i in l if i!='US'])
    
    PERSONAL_PRONOUNS.append(s)
    


# In[1061]:


# let's add feature
links['PERSONAL_PRONOUNS']=PERSONAL_PRONOUNS


# In[1063]:


links.head(2)


# # 8	Average Word Length
# Average Word Length is calculated by the formula:\
# Sum of the total number of characters in each word/Total number of words

# In[1070]:


# let's define a function 

def avg_word_len(x):
    words_len=[len(i) for i in x]
    some=sum(words_len)
    
    return round(some/len(x))


# In[1071]:


# let's add feature
links['AVG WORD LENGTH']=links.article.apply(lambda x:avg_word_len(x))


# In[1072]:


links.head()


# In[1083]:


df=links.drop(columns=['article','pos_neg'])


# In[1085]:


df.head(1)


# In[1086]:


df.shape


# In[1087]:


df.to_excel('Output Data Structure.xlsx')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




