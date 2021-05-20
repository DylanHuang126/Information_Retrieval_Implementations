import numpy as np
import pandas as pd
import os
import collections
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import csv
import tqdm

def cosine(doc1, doc2):
    return np.inner(doc1, doc2)

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

class vectorizer():
    def __init__(self):
        self.doc_nb = 1095
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
        df2 = df.set_index(['term'])
        self.dic = df2
        self.df = df
        return
    
    def tokenizer(self, doc):
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
    
    def vectorize(self, docID):
        with open('IRTM/{}.txt'.format(docID)) as infile:
            doc = infile.read().lower()
        tokens = self.tokenizer(doc)

        #calculate term frequency by Counter
        c = collections.Counter(tokens)
        tfidf_dict = {}
        for word in sorted(c):
            # tf * idf
            tf_idf = c[word] * np.log10(self.doc_nb / self.dic.loc[word].df)
            tfidf_dict[self.dic.loc[word].t_index] = tf_idf
        n = np.linalg.norm(np.array(list(tfidf_dict.values())))
        tfidf_df = pd.DataFrame(list(tfidf_dict.items()), columns=['t_index', 'tf-idf'])
        tfidf_df['tf-idf'] = tfidf_df['tf-idf'] / n
        tfidf_df = tfidf_df.set_index(['t_index'])
        tfidf_df = tfidf_df.reindex(list(range(self.df.index.min()+1,self.df.index.max()+2)),fill_value=0)
        vec = np.array(tfidf_df['tf-idf'])
        return vec
    
    def vectorize_doc(self):
        self.doc_vec = dict()
        for i in tqdm.trange(1, 1096):
            v = self.vectorize(str(i))
            self.doc_vec[i] = v

v = vectorizer()
v.vectorize_doc()

class HAC():
    def __init__(self, doc_vec, K):
        self.doc_vec = doc_vec
        self.doc_nb = len(doc_vec)
        self.K = K
        self.C = np.zeros(shape=(self.doc_nb, self.doc_nb))
        self.I = np.ones(shape=(self.doc_nb))
        for i in tqdm.trange(self.doc_nb):
            for j in range(self.doc_nb):
                self.C[i][j] = cosine(doc_vec[i + 1], doc_vec[j + 1])
        return
    
    def cluster(self):
        self.A = []
        for i in tqdm.trange(self.doc_nb - 1):
            nn = mm = 0
            max_sim = -1
            for n in range(self.doc_nb):
                for m in range(n + 1, self.doc_nb):
                    if self.I[n] and self.I[m]:
                        if self.C[n][m] > max_sim:
                            max_sim = self.C[n][m]
                            nn = n
                            mm = m
                            
            self.I[mm] = 0
            self.A.append((nn, mm))
            
            # complete link
            for j in range(self.doc_nb):
                min_sim = min(self.C[nn][j], self.C[mm][j])
                self.C[nn, j] = self.C[j, nn] = min_sim

        # merge cluster       
        centriod = []
        for docs in self.A[-(self.K - 1):]:
            for d in docs:  
                centriod.append(d)
        centriod = list(set(centriod))
        cluster = [[c] for c in centriod]
        for i in range(self.doc_nb - self.K - 1, -1, -1):
            doc1 = self.A[i][0]
            doc2 = self.A[i][1]
            for c in cluster:
                if doc1 in c:
                    c.append(doc2)
                    break
        new_c = []
        for c in cluster:
            c.sort()
            c = [i + 1 for i in c]
            new_c.append(c)
        self.c = new_c
        return

h8 = HAC(v.doc_vec, 8)
h8.cluster()
h13 = HAC(v.doc_vec, 13)
h13.cluster()
h20 = HAC(v.doc_vec, 20)
h20.cluster()

def write_result(filename, data):
    with open(filename , 'w') as outfile:
        for c in data:
            for d in c:
                outfile.write(str(d) + '\n')
            outfile.write('\n')

write_result('8.txt', h8.c)
write_result('13.txt', h13.c)
write_result('20.txt', h20.c)
