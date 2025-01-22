import json
import numpy as np
import tensorflow as tf
from model import build_lstm_model
import os
from datetime import datetime

# Load mappings and hyperparameters
with open("../data/chars.json", "r") as f:
    data = json.load(f)
    chars = data["chars"]
    char_to_idx = data["char_to_idx"]
    idx_to_char = {i: c for i, c in enumerate(chars)}

sequence_length = 100  # Must match training
vocab_size = len(chars)
embedding_dim = 256
hidden_size = 512

# Rebuild model
model = build_lstm_model(vocab_size, embedding_dim, hidden_size, sequence_length)
model.load_weights("../models/lstm_model.keras")

def generate_text(start_str, temperature=0.5, max_length=500):
    generated = list(start_str)
    input_seq = [char_to_idx[c] for c in start_str]
    
    # Pad/crop to sequence_length
    if len(input_seq) < sequence_length:
        pad_value = char_to_idx.get(' ', 0)  # Use space or default
        input_seq = [pad_value] * (sequence_length - len(input_seq)) + input_seq
    else:
        input_seq = input_seq[-sequence_length:]
    
    for _ in range(max_length):
        x = np.array(input_seq[-sequence_length:]).reshape(1, sequence_length)
        probs = model.predict(x, verbose=0)[0]
        scaled_probs = probs ** (1 / temperature)
        scaled_probs /= scaled_probs.sum()
        next_idx = np.random.choice(len(scaled_probs), p=scaled_probs)
        generated.append(idx_to_char[next_idx])
        input_seq.append(next_idx)
    
    return "".join(generated)

print(generate_text("romeo: but soft, what light through yonder window breaks? it is the east, and juliet is the sun.", temperature=0.5))

# Save to outputs/ with timestamp
os.makedirs("../outputs", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
with open(f"../outputs/generated_{timestamp}.txt", "w") as f:
    f.write(generate_text("romeo:", temperature=0.5))