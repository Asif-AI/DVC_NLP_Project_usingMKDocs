from sklearn.feature_extraction.text import CountVactorizers

corpus = [
    "zebra apple ball cat",
    "ball, cat, dog, elephant",
    "very very unique"
]

Vectorizer = CountVactorizers()
x = Vectorizer.fit_transform(corpus)
print(X.toarray())
print(vectorizer.get_feature_names_out())


max_features = 100
ngrams= 3
