import numpy as np
from gensim.models import Word2Vec

from .utils import check_nltk_package

def encode_characters_onehot(sonnets):
    """Given a list of lists, with the outer list containing sonnets and
    the inner list containing lines of sonnets (i.e. the output of
    `utils.load_shakespeare` or `utils.load_spenser`), generate character-based
    one-hot encoding.

    This function returns a dictionary encoding each character to a one-hot
    encoded vector. Each vector is a numpy array.
    """
    # Set of all characters.
    chars = set()
    for sonnet in sonnets:
        for line in sonnet:
            chars |= set(line)
    chars = list(sorted(chars))  # Needs to be sorted because sets have arbitrary order.
    n_chars = len(chars)

    encoding = {}
    for i, char in enumerate(chars):
        v = np.zeros(n_chars, dtype=int)
        v[i] = 1
        encoding[char] = v

    return encoding

def encode_words_word2vec(sonnets, size=100, window=5, *args, **kwargs):
    """Given a list of lists, with the outer list containing sonnets and
    the inner list containing lines of sonnets (i.e. the output of
    `utils.load_shakespeare` or `utils.load_spenser`), generate word-based
    Word2Vec encoding of specified size. Internally, each line is tokenized
    using `nltk.word_tokenize`. Any additional arguments are passed to the
    `Word2Vec` constructor.

    This function returns a gensim.models.keyedvectors.Word2VecKeyedVectors
    encoding each word to a vector. This object can be used like a dictionary, but
    also has useful functions such as `similar_by_vector` which can be used to find
    a word closest given an encoding.
    """
    check_nltk_package('tokenize/punkt')

    # nltk requires a list of sentences
    # each sentence is a list of words
    # We consider the new line character a "word"
    sentences = []
    for sonnet in sonnets:
        s = []
        for i, line in enumerate(sonnet):
            s.extend(nltk.word_tokenize(line))
            if i + 1 < len(sonnet):
                s.append('\n')
        sentences.append(s)

    w2v = Word2Vec(sentences, size=size, min_count=1, window=window, *args, **kwargs)
    return w2v.wv
