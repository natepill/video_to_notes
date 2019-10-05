from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, LatentDirichletAllocation


# NOTE: NMF works better on smaller corpuses where LDA is better for larger
# I'm unsure as to what constitutes a "small" vs "large" dataset... for now:
# Small dataset: <= 1 minute long video transcript
# Large dataset: > 1 minute long video transcript


# TODO: Swap out example corpus with video transcript that has been split on sentences
# to form documents.



def main_topics_small_corpus(corpus, num_topics, n_top_words):
    """
        Extract main topics from a small corpus using Non Negative Matrix Factorization (NMF)

        Args:
            corpus (array): Array of documents
            num_topics (int): Number of topics to extract
            top_words (int): Number of top words to extract from each topic

        Returns:
            {String: Array[String]}: {Topic#: Top words from each topic}

    """

    vectorizer = TfidfVectorizer(stop_words='english')

    tfidf = vectorizer.fit_transform(corpus)

    # Create NMF w/ 2 topics
    nmf = NMF(n_components=num_topics, random_state=1).fit(tfidf)
    W = nmf.fit_transform(tfidf)

    # print(W)
    feature_names = vectorizer.get_feature_names()

    topics_json = {}


    # Create dictionary of topic number to topic words
    for index, topic in enumerate(nmf.components_):
        topics_json[f"Topic {index+1}"] = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]

    # print(topics_json)

    return topics_json





def main_topics_large_corpus(corpus, num_topics, n_top_words):
    """
        Extract main topics from a large corpus using Latent Dirichlet Allocation (LDA)

        Args:
            corpus (array): Array of documents
            num_topics (int): Number of topics to extract
            top_words (int): Number of top words to extract from each topic

        Returns:
            {String: Array[String]}: {Topic#: Top words from each topic}

    """

    # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
    tf_vectorizer = CountVectorizer(stop_words='english')
    tf = tf_vectorizer.fit_transform(corpus)
    feature_names = tf_vectorizer.get_feature_names()

    # fit LDA w/ num topics and bag of words
    lda = LatentDirichletAllocation(n_components=num_topics).fit(tf)


    topics_json = {}

    # Create dictionary of topic number to topic words
    for index, topic in enumerate(lda.components_):
        topics_json[f"Topic {index+1}"] = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]

    # print(topics_json)

    return topics_json

if __name__ == "__main__":
    # Example corpus

    documents = ["I was so hungry last night, I ate all the food in the fridge",
                "The food in the restaurant tasted great, I left a huge tip",
                "The pork ribs were delicious, I'll be eating there every week.",
                "I need to use Linear Regression in order to predict Boston Housing",
                "My model classifies the emotions with a 90% accuracy",
                "My cancer dectection models needs to be optimized around a high recall score",
                "Waffles and fried chicken are a tasty combo!"]

    print(main_topics_small_corpus(documents, 2, 10))
    # print(main_topics_large_corpus(documents, 2, 10))
