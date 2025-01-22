import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
from model import SimpleRNNModel

# Full corrected data pipeline
text = open("../data/shakespeare.txt").read().lower()
chars = sorted(list(set(text)))
char_to_idx = {c: i for i, c in enumerate(chars)}
data = [char_to_idx[c] for c in text]

sequence_length = 50
step = sequence_length  # Non-overlapping sequences
X = []
y = []
for i in range(0, len(data) - sequence_length, step):
    X.append(data[i:i+sequence_length])
    y.append(data[i+sequence_length])

X = np.array(X)
y = np.array(y)

# Dataset with on-the-fly one-hot encoding
dataset = tf.data.Dataset.from_tensor_slices((X, y))
dataset = dataset.map(lambda x, y: (tf.one_hot(x, len(chars)), tf.one_hot(y, len(chars))))
dataset = dataset.batch(32).shuffle(10000)

# Initialize model
model = SimpleRNNModel(
    input_size=len(chars),
    hidden_size=128,
    output_size=len(chars)
).get_model()

# Compile and train
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(dataset, epochs=20)
model.save_weights('../models/rnn_tf.weights.h5')
model.save('../models/rnn_model.keras') 