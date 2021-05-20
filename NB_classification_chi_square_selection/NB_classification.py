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
    new_token_stem = []
    for t in token_stem:
        if t not in sw and len(t) > 2:
            new_token_stem.append(t)
    return new_token_stem

train_dict = dict()
with open('train.txt') as infile:
    f = infile.readlines()
    for c in f:
        l = c.split()
        train_dict[l[0]] = l[1:]

N = 0
C = len(train_dict)
prior = np.empty(shape=(C, 1))
docs = ''
for idx, classs in enumerate(train_dict.keys()):
    
    # prior prob
    prior[idx] = len(train_dict[classs])
    N += len(train_dict[classs])
    
    # extract vocabulary
    for c in train_dict[classs]:
        with open('IRTM/{}.txt'.format(c)) as infile:
            doc = infile.read().lower()
            docs += doc + ' '
V = set(tokenizer(docs))
prior = prior / N

#feature selection using chi-square score
feature_dict = dict()
for idx, classs in enumerate(train_dict.keys()):
    tokens_list = []
    for c in train_dict[classs]:
        with open("IRTM/{}.txt".format(c)) as infile:
            doc = infile.read().lower()
            tokens_list.append(tokenizer(doc))
    feature_dict[idx] = tokens_list

chi_dict = dict()
for idx, classs in enumerate(train_dict.keys()):
    rel = feature_dict[idx]
    irr = [feature_dict[i] for i,_ in enumerate(train_dict.keys()) if i != idx]
    
    for v in V:
        pont = pofft = 0
        for doc in rel:
            for w in doc:
                if v == w:
                    pont += 1
                    break
        for c in irr:
            for doc in c:
                for iw in doc:
                    if v == iw:
                        pofft += 1
                        break

        aont = len(rel) - pont
        aofft = np.sum([len(i) for i in irr]) - pofft
        tol = len(rel) + np.sum([len(i) for i in irr])
        
        epont = (pont + aont) * (pont + pofft) / tol
        eaont = (pont + aont) * (aont + aofft) / tol
        epofft = (pofft + aofft) * (pont + pofft) / tol
        eaofft = (pofft + aofft) * (aont + aofft) / tol
        
        v_chi = (pont - epont)**2 / epont + (aont - eaont)**2 / eaont + (pofft - epofft)**2 / epofft + (aofft - eaofft)**2 / eaofft
        
        if idx == 0:
            chi_dict[v] = dict()
        chi_dict[v][idx] = v_chi

chi = dict()
for v in chi_dict.keys():
    s = 0
    for c in chi_dict[v].keys():
        s += chi_dict[v][c]
    s /= 13
    chi[v] = s
chi_sorted = sorted(chi.items(), key=lambda k: k[1])
chi_sorted.reverse()
features = set([chi_sorted[i][0] for i in range(500)])


condprob = np.empty(shape=(len(features), C))
t_id = dict()
for idx, classs in enumerate(train_dict.keys()): # 13 classes
    docs = ''
    for c in train_dict[classs]:
        with open('IRTM/{}.txt'.format(c)) as infile:
            doc = infile.read().lower()
            docs += doc + ' '
    VV = tokenizer(docs)
    for i, t in enumerate(features):
        count = 0
        for v in VV:
            if t == v:
                count += 1
        condprob[i][idx] = count
        t_id[t] = i
    for i, t in enumerate(features):
        condprob[i][idx] = condprob[i][idx] + 1 / (len(VV) + len(features))

train = []
alldocs = [i for i in range(1, 1096)]
for classs in train_dict.keys():
    for d in train_dict[classs]:
        train.append(int(d))
test = set(alldocs) - set(train)
test = list(test)

# generate output
with open("output_f2.csv", 'w', newline = '') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['Id', 'Value'])
    for idx in test:
        score = np.empty(shape=(C, 1))
        with open("IRTM/{}.txt".format(str(idx))) as infile:
            doc = infile.read().lower()
            W = tokenizer(doc)
            new_W = []
            for w in W:
                if w in features:
                    new_W.append(w)
        for i, classs in enumerate(train_dict.keys()):
            score[i] = prior[i]
            for w in new_W:
                score[i] += np.log(condprob[t_id[w]][i])
        writer.writerow([idx, np.argmax(score)+1])
