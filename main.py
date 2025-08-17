# movie_sentiment_dl.py

import pandas as pd
import numpy as np
import re
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# NLTK for text cleaning
import nltk
from nltk.corpus import stopwords

# Scikit-learn for splitting data and evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# TensorFlow / Keras for Deep Learning
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# --- Configuration ---
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Model & Tokenizer Parameters
VOCAB_SIZE = 10000
MAX_LENGTH = 200
EMBEDDING_DIM = 128

# --- Step 1: Load Data ---
print("1. Loading and preparing data...")
try:
    df = pd.read_csv("test_data.csv")
    # Renaming columns 
    if df.shape[1] == 2:
        df.columns = ['review', 'sentiment']
except FileNotFoundError:
    print("Error: 'test_data.csv' not found.")
    exit()

if df['sentiment'].dtype == object:
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

print("Sentiment unique values:", df['sentiment'].unique())

# --- Text Cleaning ---
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df['review'] = df['review'].astype(str).apply(clean_text)

# --- Step 2: Split Data ---
print("2. Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['sentiment'],
    test_size=0.2,
    random_state=42,
    stratify=df['sentiment']  # ensures class balance
)

# --- Step 3: Tokenization & Padding ---
print("3. Tokenizing and padding text...")
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LENGTH, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LENGTH, padding='post', truncating='post')

# --- Step 4: Build Model ---
print("4. Building model...")
model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LENGTH),
    GlobalAveragePooling1D(),
    Dense(24, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# --- Step 5: Train Model ---
print("5. Training model...")
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train_pad, y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1
)

# --- Step 6: Evaluate Model ---
print("6. Evaluating model...")
y_pred_prob = model.predict(X_test_pad)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Negative','Positive']))

# --- Step 7: Confusion Matrix ---
print("7. Generating confusion matrix...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative','Positive'],
            yticklabels=['Negative','Positive'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix_dl.png")
plt.show()
print("Saved confusion matrix as 'confusion_matrix_dl.png'")

# --- Step 8: Save Model & Tokenizer ---
print("8. Saving model and tokenizer...")
model.save('sentiment_model.h5')
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
print("Model saved as 'sentiment_model.h5' and tokenizer as 'tokenizer.pkl'.")
