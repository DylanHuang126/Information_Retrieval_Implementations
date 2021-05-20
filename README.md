# Information_Retrieval_Implementations
A series of Information Retrieval and Text Mining technique implementation.

## HAC Clustering

### Usage
程式會讀入在同一資料夾內路徑為”IRTM”的資料夾，並為裡面所有的檔案分群。

### Dataset

English news document collections.

### Requirements
* python 3.7
* numpy
* pandas
* tqdm
* nltk
* re





## Multinomial Naive Bayes Classifier with chi-square feature selection

### Usage
程式會讀入 train.txt 當作 ground true data，並分類 ”IRTM” 資料夾內的 news documents，最終會產出 csv 檔案做為分類結果。

### Dataset
English news document collections.

### Requirements
* python 3.7
* numpy
* pandas
* nltk




## Term Frequency and Inverse Document Frequency
### Usage
程式會讀入在同一資料夾內路徑為 ”IRTM” 的資料夾，並開啟裡面所有的檔案作為 tf-idf 計算的依據，程式最終會產出: “dictionary.txt”, “1.txt” ，並計算出 1.txt 與 2.txt 的 Cosine Similarity。

### Dataset
English news document collections.

### Requirements
* python 3.7
* numpy
* pandas
* nltk




## Porter Stemming
### Requirements
* python 3.7
* nltk
* re


### Dataset

Any English news document.

### Usage

程式會讀入在同一資料夾內的”doc.txt”，並產生出”new_doc.txt”。 new_doc.txt 中會列出所有 stemming 過後且不包含 stopwords 的單字。

