import numpy as np
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

    def generate(self, seed, char_to_dim, dim_to_char):
        X = np.zeros((len(seed), self.n_chars))
        indices = np.vstack((np.arange(len(seed)), [char_to_dim[char] for char in seed]))
        X[tuple(indices)] = 1
        generated = seed

        # Loop until we have 14 lines.
        while generated.count('\n') < 14:
            probabilities = self.predict(np.expand_dims(X, 0)).flatten()
            #choice = np.argmax(probabilities)
            choice = np.random.choice(np.arange(self.n_chars), p=probabilities)
            generated += dim_to_char[choice]

            # "Push back" X by one and add new prediction to the end.
            X[:X.shape[0]-1] = X[1:]
            X[-1] = 0
            X[-1][choice] = 1
        return generated
