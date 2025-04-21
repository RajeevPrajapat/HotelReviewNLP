from flask import Flask, render_template, request
import pickle
import re
import nltk
import pycountry
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK stopwords (only runs once)
nltk.download('stopwords')

# Load sentiment model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

app = Flask(__name__)

# Initialize pre-processing tools
ps = PorterStemmer()
stop_words = set(stopwords.words("english")) - {'no', 'nor', 'not'}
country_set = {country.name.lower() for country in pycountry.countries}


# 🧼 Function to clean and preprocess text
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# 🌍 Function to detect country name and highlight it
def detect_country(original_text):
    words = original_text.split()
    highlighted = []
    found_country = None

    for word in words:
        word_clean = re.sub(r'[^\w\s]', '', word).lower()
        if word_clean in country_set:
            found_country = word.title()
            word = f"<span style='color:blue;font-weight:bold'>{word}</span>"
        highlighted.append(word)

    return ' '.join(highlighted), found_country

# 🚫 Function to validate the review (not just gibberish)
def is_valid_review(text):
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
    return len(words) >= 3


# 🔄 ROUTES

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        review = request.form["review"]

        # Validate review
        if not is_valid_review(review):
            return render_template(
                "index.html",
                prediction=None,
                review_highlight=None,
                country=None,
                error="❌ Invalid review. Please enter a meaningful hotel experience."
            )

        # Preprocess and predict
        cleaned = clean_text(review)
        vectorized = tfidf.transform([cleaned]).toarray()
        pred = model.predict(vectorized)[0]

        # Sentiment label
        label_map = {0: "Negative 😠", 1: "Neutral 😐", 2: "Positive 😊"}
        prediction_text = label_map[pred]

        # Detect country
        highlighted_review, found_country = detect_country(review)

        return render_template(
            "index.html",
            prediction=prediction_text,
            review_highlight=highlighted_review,
            country=found_country,
            error=None
        )


if __name__ == "__main__":
    app.run(debug=True)
