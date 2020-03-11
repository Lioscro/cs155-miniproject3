import nltk
import numpy as np
from gensim.models import Word2Vec

from .utils import check_nltk_package

def encode_characters_onehot(sonnets):
    """Given a list of lists, with the outer list containing sonnets and
    the inner list containing lines of sonnets (i.e. the output of
    `utils.load_shakespeare` or `utils.load_spenser`), generate character-based
    one-hot encoding. All characters are lowercased.

    This function returns a tuple of two elements.
    The first is a dictionary encoding each character to a one-hot
    encoded vector. Each vector is a numpy array. The second is a list of encoded
    sonnets.
    """
    # Set of all characters.
    chars = set()
    for sonnet in sonnets:
        for line in sonnet:
            chars |= set(line.lower())
    chars = list(sorted(chars))  # Needs to be sorted because sets have arbitrary order.
    n_chars = len(chars)

    encoding = {}
    for i, char in enumerate(chars):
        v = np.zeros(n_chars, dtype=int)
        v[i] = 1
        encoding[char] = v

    # Encode the sonnets.
    encoded_sonnets = []
    for sonnet in sonnets:
        encoded = []
        for i, line in enumerate(sonnet):
            encoded.extend([encoding[char] for char in line])
            encoded.append(encoding['\n'])
        encoded_sonnets.append(encoded)

    return encoding, encoded_sonnets

def encode_words_word2vec(sonnets, size=100, window=5, *args, **kwargs):
    """Given a list of lists, with the outer list containing sonnets and
    the inner list containing lines of sonnets (i.e. the output of
    `utils.load_shakespeare` or `utils.load_spenser`), generate word-based
    Word2Vec encoding of specified size. Internally, each line is tokenized
    using `nltk.word_tokenize`. Any additional arguments are passed to the
    `Word2Vec` constructor. All characters are lowercased.

    This function returns a tuple of two elements.
    The first is a gensim.models.keyedvectors.Word2VecKeyedVectors
    encoding each word to a vector. This object can be used like a dictionary, but
    also has useful functions such as `similar_by_vector` which can be used to find
    a word closest given an encoding. The second is a list of encoded sonnets.
    """
    nltk.download('punkt')

    # nltk requires a list of sentences
    # each sentence is a list of words
    # We consider the new line character a "word"
    sentences = []
    for sonnet in sonnets:
        s = []
        for i, line in enumerate(sonnet):
            s.extend(nltk.word_tokenize(line.lower()))
            if i + 1 < len(sonnet):
                s.append('\n')
        sentences.append(s)

    w2v = Word2Vec(sentences, size=size, min_count=1, window=window, *args, **kwargs)
    encoding = w2v.wv

    # Encode.
    encoded_sonnets = []
    for sonnet in sentences:
        encoded_sonnets.append([encoding[word] for word in sonnet])

    return w2v.wv, encoded_sonnets
