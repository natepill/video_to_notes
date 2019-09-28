from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, LatentDirichletAllocation


# NOTE: NMF works better on smaller corpuses where LDA is better for larger
# I'm unsure as to what constitutes a "small" vs "large" dataset... for now:
# Small dataset: <= 1 minute long video transcript
# Large dataset: > 1 minute long video transcript


# TODO: Swap out example corpus with video transcript that has been split on sentences
# to form documents.

# Example corpus
documents = ["This little kitty came to play when I was eating at a restaurant.",
             "Merley has the best squooshy kitten belly.",
             "Google Translate app is incredible.",
             "If you open 100 tab in google you get a smiley face.",
             "Best cat photo I've ever taken.",
             "Climbing ninja cat.",
             "Impressed with google map feedback.",
             "Key promoter extension for Google Chrome."]


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

    tfidf = vectorizer.fit_transform(documents)

    # Create NMF w/ 2 topics
    num_topics = 2
    nmf = NMF(n_components=num_topics, random_state=1).fit(tfidf)
    W = nmf.fit_transform(tfidf)

    # print(W)

    n_top_words = 10
    feature_names = vectorizer.get_feature_names()

    for index, topic in enumerate(nmf.components_):
        print(f"Topic Group {index}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words -1:-1]]))



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
tf = tf_vectorizer.fit_transform(documents)
tf_feature_names = tf_vectorizer.get_feature_names()
