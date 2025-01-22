import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model

def build_lstm_model(vocab_size, embedding_dim, hidden_size, sequence_length):
    # Define input layer with explicit sequence length
    inputs = Input(shape=(sequence_length,), name="input_layer")
    
    # Embedding layer
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs)
    
    # LSTM layer
    lstm_out, state_h, state_c = LSTM(
        hidden_size, 
        return_sequences=False,  # Critical for single-output-per-sequence
        return_state=True, 
        name="lstm_layer"
    )(x)
    
    # Output layer
    outputs = Dense(vocab_size, activation="softmax")(lstm_out)
    
    # Define the model
    model = Model(
        inputs=inputs, 
        outputs=outputs, 
        name="lstm_text_generator"
    )
    
    return model