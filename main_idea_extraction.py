# nltk.download('stopwords')
# nltk.download('punkt') # one time execution

import pandas as pd
import numpy as np
import networkx as nx
import nltk
import pickle
import re
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

# NOTE: For local testing
# df = pd.read_csv("tennis_articles_v4.csv")


def main_ideas(words):
    sentences = []
    # for s in df["article_text"]:
    for s in words:
        # print("S:", s)
        sentences.append(sent_tokenize(s))

    # flatten list
    sentences = [y for x in sentences for y in x]

    word_embeddings_file = open('glove_word_embeddings','rb')
    word_embeddings = pickle.load(word_embeddings_file)
    word_embeddings_file.close()

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



    # similarity matrix that will contain the cosine similarity scores of the sentences.
    sim_mat = np.zeros([len(sentences), len(sentences)])


    # Initialize the matrix with cosine similarity scores.
    for i in range(len(sentences)):
      for j in range(len(sentences)):
        # NOTE: Clarify what i == j means in terms of our similarity matrix
        # NOTE: Why do we need to reshape our vectors, I thought they are already (1,100)?
        if i != j:
          sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]

    # The nodes of this graph will represent the sentences and the edges
    # will represent the similarity scores between the sentences. On this graph,
    # we will apply the PageRank algorithm to arrive at the sentence rankings.
    nx_graph = nx.from_numpy_array(sim_mat)
    # Apply pagerank algorithm since its the same as TextRank
    scores = nx.pagerank(nx_graph)

    # Extract the top sentences for summary extraction
    ranked_sentences = sorted(((scores[i], s) for i,s in enumerate(sentences)), reverse=True)
    return ranked_sentences

    # Print Top 10 sentences
    #for i in range(10):
        #print(ranked_sentences[i][1])

# if __name__ == "__main__":
#     main_ideas(df)
    # print(main_ideas(df))
    # print(len(df["article_text"]))


#
# if __name__ == "__main__":
#     word_embeddings = {}
#     f = open('glove.6B/glove.6B.100d.txt', encoding='utf-8')
#     for line in f:
#         values = line.split()
#         word = values[0]
#         coefs = np.asarray(values[1:], dtype='float32')
#         word_embeddings[word] = coefs
#     f.close()
#
#     word_embeddings_file = f = open('glove_word_embeddings', 'wb')
#     pickle.dump(word_embeddings, word_embeddings_file)
#     word_embeddings_file.close()
