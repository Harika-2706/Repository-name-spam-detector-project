from flask import Flask, request, jsonify, render_template
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# 🔹 Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', ' url ', text)   # URLs
    text = re.sub(r'\d+', ' number ', text)    # numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)   # remove special chars
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 🔹 Load dataset
data = pd.read_csv("spam.csv", encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Clean messages
data['message'] = data['message'].apply(clean_text)

# Convert labels
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2
)

# Build model
model = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words='english', ngram_range=(1,2))),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Train model
model.fit(X_train, y_train)

# ---------------- UI ROUTE ----------------
@app.route('/')
def home():
    return render_template('index.html')

# ---------------- PREDICT API ----------------
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = clean_text(data['message'])

    # 🔥 Get probability
    proba = model.predict_proba([message])[0]
    spam_prob = proba[1] * 100
    spam_keywords = ['free', 'win', 'winner', 'money', 'prize', 'urgent', 'claim']
    keyword_flag = any(word in message for word in spam_keywords)

    # Decision
    if spam_prob > 50 or keyword_flag:
        result = "Spam"
    else:
        result = "Not Spam"

    return jsonify({
        'result': result,
        'confidence': round(spam_prob, 2)
    })

# ---------------- REPORT API ----------------
@app.route('/report', methods=['POST'])
def report():
    data = request.get_json()
    message = data['message']

    with open("reported_spam.txt", "a", encoding="utf-8") as f:
        f.write(message + "\n")

    return jsonify({"status": "Reported successfully"})
@app.route('/history')
def history():
    try:
        with open("reported_spam.txt", "r", encoding="utf-8") as f:
            messages = f.readlines()
    except FileNotFoundError:
        messages = []

    return render_template("history.html", messages=messages)
# ---------------- RUN SERVER ----------------
if __name__ == '__main__':
    app.run(debug=True)