nltk.download('stopwords')
nltk.download('punkt') # one time execution

import numpy as np
import pandas as pd
import nltk
import re
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords


df = pd.read_csv("tennis_articles_v4.csv")


sentences = []
for s in df["article_text"]:
    sentences.append(sent_tokenize(s))

# flatten list
sentences = [y for x in sentences for y in x]


word_embeddings = {}
f = open('glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()

# remove punctuations, numbers and special characters
clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

# make alphabets lowercase
clean_sentences = [s.lower() for s in clean_sentences]


stop_words = stopwords.words('english')

# helper function to remove stopwords
def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new

# remove stopwords from the sentences
clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]


# We will first fetch vectors (each of size 100 elements) for the constituent words
# in a sentence and then take mean/average of those vectors to arrive at a
# consolidated vector for the sentence.

sentence_vectors = []
for sentence in clean_sentences:
    if len(sentence) != 0:
        vector = sum([word_embeddings.get(w, np.zeros((100,))) for w in sentence.split()])/(len(sentence.split())+0.001)
    else:
        vector = np.zeros((100,))
    sentence_vectors.append(vector)



# similarity matrix
sim_mat = np.zeros([len(sentences), len(sentences)])
sim_mat = np.zeros([len(sentences), len(sentences)])
