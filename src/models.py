from keras.layers import Activation, Dense, Lambda, LSTM
from keras.models import Sequential

from src import utils

class CharacterLSTM(Sequential):
    def __init__(self, units, window_size, n_chars, temperature=1, *args, **kwargs):
        super(CharacterLSTM, self).__init__(*args, **kwargs)

        # Add layers
        self.add(LSTM(
            units,
            input_shape=(window_size, n_chars)
        ))
        self.add(Dense(n_chars))
        self.add(Lambda(lambda x: x / temperature))
        self.add(Activation('softmax'))
