import numpy as np
import pandas as pd
import os
import collections
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

#--------hw1--------#
def tokenizer(doc):
    s = re.findall(r"[a-z]+", doc)
    #Stemming: using Porter's algorithm in nltk stemming library
    token_stem = []
    ps = PorterStemmer()
    for t in s:
        token_stem.append(ps.stem(t))
    
    #Stopword removal: using english stopwords in nltk
    sw = set(stopwords.words('english'))
    for t in token_stem:
        if t in sw:
            token_stem.remove(t)
    return token_stem

#--------part1--------#
term_dict = {}
for txt in os.listdir('IRTM'):
    if 'txt' in txt:
        with open('IRTM/' + txt) as infile:
            doc = infile.read().lower()
        tokens = set(tokenizer(doc))#remove duplicate token
        
        #put token into dictionary and add up its occurrence
        for t in tokens:
            if t not in term_dict.keys():
                term_dict[t] = 1
            else:
                term_dict[t] += 1

# sort term dictionary
od = collections.OrderedDict(sorted(term_dict.items()))
df = pd.DataFrame(list(od.items()), columns=['term', 'df'])
df['t_index'] = np.arange(1, len(df) + 1)
df = df[['t_index', 'term', 'df']]
df.to_csv('dictionary.txt', index=False, sep=' ', header="t_index\tterm\tdf")

#--------part2--------#
df2 = df.set_index(['term'])
doc_nb = 1095

with open('IRTM/1.txt') as infile:
    doc = infile.read().lower()
tokens = tokenizer(doc)

#calculate term frequency by Counter
c = collections.Counter(tokens)
term_nb = len(c)
tfidf_dict = {}
for word in sorted(c):
    # tf * idf
    tf_idf = c[word] * np.log10(doc_nb / df2.loc[word].df)
    tfidf_dict[df2.loc[word].t_index] = tf_idf
n = np.linalg.norm(np.array(list(tfidf_dict.values())))
tfidf_df = pd.DataFrame(list(tfidf_dict.items()), columns=['t_index', 'tf-idf'])
tfidf_df['tf-idf'] = tfidf_df['tf-idf'] / n
tfidf_df.to_csv('1.txt', index=False, sep=' ', header="t_index\ttf-idf")

#write number of terms in first line
with open('1.txt', 'r+') as file:
    doc = file.read()
    with open('1.txt', 'w+') as newfile:
        newfile.write(str(term_nb) + '\n')
        newfile.write(doc)

#--------part3--------#
with open('IRTM/3.txt') as infile:
    doc = infile.read().lower()
tokens = tokenizer(doc)

c = collections.Counter(tokens)
term_nb2 = len(c)
tfidf_dict2 = {}
for word in sorted(c):
    tf_idf = c[word] * np.log10(doc_nb / df2.loc[word].df)
    tfidf_dict2[df2.loc[word].t_index] = tf_idf
n = np.linalg.norm(np.array(list(tfidf_dict2.values())))
tfidf_df2 = pd.DataFrame(list(tfidf_dict2.items()), columns=['t_index', 'tf-idf'])
tfidf_df2['tf-idf'] = tfidf_df2['tf-idf'] / n

#fill missing index and value, tranform them into vector
tfidf_df = tfidf_df.set_index(['t_index'])
tfidf_df = tfidf_df.reindex(list(range(df.index.min()+1,df.index.max()+2)),fill_value=0)
tfidf_df2 = tfidf_df2.set_index(['t_index'])
tfidf_df2 = tfidf_df2.reindex(list(range(df.index.min()+1,df.index.max()+2)),fill_value=0)
vec1 = np.array(tfidf_df['tf-idf'])
vec2 = np.array(tfidf_df2['tf-idf'])

def cosine(doc1, doc2):
    return np.inner(doc1, doc2)

print(cosine(vec1, vec2))
