import pickle

vectorizer, model = pickle.load(open('bank_model.pkl', 'rb'))

text = ["I lost my card"]
X = vectorizer.transform(text)
print(model.predict(X))