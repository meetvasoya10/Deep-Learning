from model import build_lstm_model
import numpy as np
import json
import os

# Load and preprocess data
text = open("../data/shakespeare.txt").read().lower()
chars = sorted(list(set(text)))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

# Save mappings for generate.py
os.makedirs("../data", exist_ok=True)
with open("../data/chars.json", "w") as f:
    json.dump({"chars": chars, "char_to_idx": char_to_idx}, f)

# Convert text to indices
data = [char_to_idx[c] for c in text]

# Create sequences and targets
sequence_length = 100  # Match this in generate.py
X = []
y = []
for i in range(0, len(data) - sequence_length, sequence_length):
    X.append(data[i:i+sequence_length])
    y.append(data[i+sequence_length])

X = np.array(X)
y = np.array(y)

# Build model
vocab_size = len(chars)
embedding_dim = 256
hidden_size = 512

model = build_lstm_model(vocab_size, embedding_dim, hidden_size, sequence_length)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",  # For integer targets
    metrics=["accuracy"]
)

# Train
model.fit(X, y, batch_size=64, epochs=20)

# Save
os.makedirs("../models", exist_ok=True)
model.save("../models/lstm_model.keras")