from flask import Flask, render_template, request
import pickle
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

# Load the trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

app = Flask(__name__)

ps = PorterStemmer()
stop_words = set(stopwords.words("english")) - {'no', 'nor', 'not'}

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        review = request.form["review"]
        cleaned = clean_text(review)
        vectorized = tfidf.transform([cleaned]).toarray()
        pred = model.predict(vectorized)[0]

        label_map = {0: "Negative 😠", 1: "Neutral 😐", 2: "Positive 😊"}
        return render_template("index.html", prediction=label_map[pred])

if __name__ == "__main__":
    app.run(debug=True)
