from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF

# Example corpus
documents = ["This little kitty came to play when I was eating at a restaurant.",
             "Merley has the best squooshy kitten belly.",
             "Google Translate app is incredible.",
             "If you open 100 tab in google you get a smiley face.",
             "Best cat photo I've ever taken.",
             "Climbing ninja cat.",
             "Impressed with google map feedback.",
             "Key promoter extension for Google Chrome."]

vectorizer = TfidfVectorizer(stop_words='english')

tfidf = vectorizer.fit_transform(documents)
print(vectorizer.get_feature_names())

# Create NMF w/ 2 topics
n_topics = 2
nmf = NMF(n_components=n_topics, random_state=1).fit(tfidf)
W = nmf.fit_transform(tfidf)

print(W)
