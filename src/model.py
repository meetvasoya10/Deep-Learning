import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.layers import SimpleRNN, Dense, Input
from tensorflow.keras.models import Model

class SimpleRNNModel:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.model = self._build_model()

    def _build_model(self):
        inputs = Input(shape=(None, self.input_size))  # (batch, seq_len, input_size)
        x = SimpleRNN(self.hidden_size, return_sequences=False)(inputs)
        outputs = Dense(self.output_size, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        return model

    def get_model(self):
        return self.model