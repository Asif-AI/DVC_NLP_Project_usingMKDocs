from sklearn.feature_extraction.text import CountVactorizer


corpus = [
    "apple ball cat",
    "ball cat dog elephant",
]

Vectorizer = CountVactorizer()
x = Vectorizer.fit_transform(corpus)
#print(X.toarray())
#print(vectorizer.get_feature_names_out())


max_features = 4
ngrams= 2

vectorizer = CountVactorizer(max_features=max_features, ngram_range=(1, ngrams))
X = vectorizer.fit_transform(corpus)
print(X.toarray())
print(vectorizer.get_feature_names_out())
