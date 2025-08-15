# demo_dl.py

import pickle
import re
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Configuration & Loading ---
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

MAX_LENGTH = 200  # Must match training

# Load model and tokenizer
try:
    model = load_model('sentiment_model.h5')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
except Exception as e:
    print("Error: Model or tokenizer not found. Run 'movie_sentiment_dl.py' first.")
    print("Details:", e)
    exit()

# --- Preprocessing Function ---
def clean_and_prepare_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = [w for w in text.split() if w not in stop_words]
    cleaned_text = " ".join(words)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LENGTH, padding='post', truncating='post')
    return padded_sequence

# --- Interactive Demo ---
print("🎬 Movie Review Sentiment Prediction Demo")
print("Type a review or 'exit' to quit.")

while True:
    user_input = input("\nEnter a movie review: ").strip()
    if user_input.lower() == 'exit':
        break
    prepared_input = clean_and_prepare_text(user_input)
    # suppress verbose output during prediction
    prediction_prob = model.predict(prepared_input, verbose=0)[0][0]
    sentiment = "Positive" if prediction_prob > 0.5 else "Negative"
    print(f"🧠 Predicted Sentiment: {sentiment} (Confidence: {prediction_prob:.2%})")

print("\nThank you for using the demo. Goodbye! 👋")
