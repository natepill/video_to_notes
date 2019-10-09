from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def tf_idf(corpus, main_topics, num_words = 5):
    tfidf = TfidfVectorizer(stop_words='english')

    tfs = tfidf.fit_transform(corpus)
    response = tfidf.transform(main_topics)
    feature_names = tfidf.get_feature_names()

    feature_array = np.array(tfidf.get_feature_names())
    tfidf_sorting = np.argsort(response.toarray()).flatten()[::-1]

    top_n = feature_array[tfidf_sorting][:num_words]

    return list(top_n)
