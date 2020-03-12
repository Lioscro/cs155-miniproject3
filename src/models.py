import numpy as np
from keras.layers import Activation, Dense, Dropout, Embedding, Lambda, LSTM
from keras.models import Sequential
from tqdm import tqdm

from src import utils

class CharacterLSTM(Sequential):
    """Character-based LSTM.
    """
    def __init__(self, units, window_size, n_chars, temperature=1, *args, **kwargs):
        super(CharacterLSTM, self).__init__(*args, **kwargs)

        self.units = units
        self.window_size = window_size
        self.n_chars = n_chars
        self.temperature = temperature

        # Add layers
        self.add(LSTM(
            units,
            input_shape=(window_size, n_chars)
        ))
        self.add(Dense(n_chars))
        self.add(Lambda(lambda x: x / temperature))
        self.add(Activation('softmax'))

    def generate(self, seed, encodings, n_lines=14):
        """Generate an `n_lines`-line sonnet.
        """
        # Decoding dictionary.
        decoding = {np.argmax(x): char for char, x in encodings.items()}
        
        n_chars = len(encodings)
        X = np.array([encodings[char] for char in seed])
        generated = seed

        # Loop until we have `n_lines` lines.
        pbar = tqdm(total=n_lines)
        pbar.update(1)
        count = generated.count('\n')
        while count < n_lines:
            probabilities = self.predict(np.expand_dims(X, 0)).flatten()
            
            choice = np.random.choice(np.arange(self.n_chars), p=probabilities)
            generated += decoding[choice]

            # "Push back" X by one and add new prediction to the end.
            X[:X.shape[0]-1] = X[1:]
            X[-1] = 0
            X[-1][choice] = 1

            # Update progress bar
            pbar.update(generated.count('\n') - count)
            count = generated.count('\n')
        pbar.close()
        return generated

class EmbeddingLSTM(Sequential):
    """Word-based LSTM that also learns word embeddings.
    """
    def __init__(self, units, window_size, n_words, size, temperature=1, *args, **kwargs):
        super(EmbeddingLSTM, self).__init__(*args, **kwargs)

        self.units = units
        self.window_size = window_size
        self.n_words = n_words
        self.size = size
        self.temperature = temperature
        
        self.add(Embedding(n_words, size, input_length=window_size))
        self.add(LSTM(units, input_shape=(window_size, size), dropout=0.5))
        self.add(Dense(n_words))
        self.add(Lambda(lambda x: x / temperature))
        self.add(Activation('softmax'))

    def generate(self, seed, encoding, n_lines=14):
        """Generate an `n_lines`-line sonnet.
        Note that the `seed` must be a list of words and `encoding` is a dictionary.
        """
        decoding = {np.argmax(x): char for char, x in encoding.items()}
        n_words = len(encoding)
        X = np.array([np.argmax(encoding[word]) for word in seed])

        generated = seed.copy()

        # Loop until we have `n_lines` lines.
        pbar = tqdm(total=n_lines)
        count = generated.count('\n')
        pbar.update(count)
        while count < n_lines:
            probabilities = self.predict(np.expand_dims(X, 0)).flatten()
            
            # Choose next word.
            choice = np.random.choice(np.arange(n_words), p=probabilities)
            generated.append(decoding[choice])

            # "Push back" X by one and add new prediction to the end.
            X[:X.shape[0]-1] = X[1:]
            X[-1] = choice

            # Update progress bar
            pbar.update(generated.count('\n') - count)
            count = generated.count('\n')
        pbar.close()
        return generated

    
class WordLSTM(Sequential):
    """Word-based LSTM to be used with words that are already embedded.
    """
    def __init__(self, units, window_size, size, temperature=1, *args, **kwargs):
        super(WordLSTM, self).__init__(*args, **kwargs)

        self.units = units
        self.window_size = window_size
        self.size = size
        self.temperature = temperature

        # Add layers
        self.add(LSTM(
            units,
            input_shape=(window_size, size)
        ))
        self.add(Dense(size))
        self.add(Lambda(lambda x: x / temperature))
        self.add(Activation('softmax'))

    def generate(self, seed, encoding, n_lines=14):
        """Generate an `n_lines`-line sonnet.
        Note that the `seed` must be a list of words and `encodings` is a
        word2vec object.
        """
        n_words = len(encoding.vocab)
        X = np.array([encoding[word] for word in seed])
        generated = seed.copy()

        # Loop until we have `n_lines` lines.
        pbar = tqdm(total=n_lines)
        pbar.update(1)
        count = generated.count('\n')
        while count < n_lines:
            predicted = self.predict(np.expand_dims(X, 0)).flatten()
            
            # Find word that is closest to predicted encoding.
            choice = encoding.similar_by_vector(predicted, topn=1)[0][0]
            generated.append(choice)

            # "Push back" X by one and add new prediction to the end.
            X[:X.shape[0]-1] = X[1:]
            X[-1] = encoding[choice]

            # Update progress bar
            pbar.update(generated.count('\n') - count)
            count = generated.count('\n')
        pbar.close()
        return generated
