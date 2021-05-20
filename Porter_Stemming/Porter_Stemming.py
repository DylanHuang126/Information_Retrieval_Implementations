import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
with open('doc.txt', 'r') as infile:
    #Lowercasing: apply lower()
    doc = infile.read().lower()

#Tokenization: remove commas and dots then split by space
doc = doc.replace(".", "").replace(",", "")
tokens = doc.split()

#Stemming: using Porter's algorithm in nltk stemming library
token_stem = []
ps = PorterStemmer()
for t in tokens:
    token_stem.append(ps.stem(t))

#Stopword removal: using english stopwords in nltk
sw = set(stopwords.words('english'))
for t in token_stem:
    if t in sw:
        token_stem.remove(t)
        
#save file: save output file
new_doc = " ".join(token_stem)
with open('new_doc.txt', 'w') as outfile:
    outfile.write(new_doc)
