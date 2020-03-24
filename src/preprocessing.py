import nltk
import numpy as np
from gensim.models import Word2Vec
import re

from .utils import check_nltk_package

def encode_characters_onehot(sonnets):
    """Given a list of lists, with the outer list containing sonnets and
    the inner list containing lines of sonnets (i.e. the output of
    `utils.load_shakespeare` or `utils.load_spenser`), generate character-based
    one-hot encoding.
    
    Preprocessing
    -------------
    - All characters are lowercased.
    - The newline character `\n` is considered a word.

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
    chars.add('\n')
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
            encoded.extend([encoding[char] for char in line.lower()])
            encoded.append(encoding['\n'])
        encoded_sonnets.append(np.array(encoded, dtype=int))

    return encoding, encoded_sonnets

def encode_words_onehot(sonnets):
    """Given a list of lists, with the outer list containing sonnets and
    the inner list containing lines of sonnets (i.e. the output of
    `utils.load_shakespeare` or `utils.load_spenser`), generate word-based
    one-hot encoding. Internally, each line is tokenized
    using `nltk.word_tokenize`.
    
    Preprocessing
    -------------
    - All characters are lowercased.
    - All punctuation and special characters are removed (,.?!;:()).
    - The newline character `\n` is considered a word.

    This function returns a tuple of two elements.
    The first is a dictionary encoding each word to a one-hot
    encoded vector. Each vector is a numpy array. The second is a list of encoded
    sonnets.
    """
    remove = ',.?!:;()'
    
    # Remove all punctuation and special characters.
    processed = []
    for sonnet in sonnets:
        p = []
        for line in sonnet:
            l = line.lower()
            for char in remove:
                l = l.replace(char, '')
            p.append(l)
        processed.append(p)
    sonnets = processed
    
    # Set of all words.
    words = set()
    for sonnet in sonnets:
        for line in sonnet:
            words |= set(nltk.word_tokenize(line))
    words.add('\n')  # Consider new line as a word.
    words = list(sorted(words))  # Needs to be sorted because sets have arbitrary order.
    n_words = len(words)

    encoding = {}
    for i, word in enumerate(words):
        v = np.zeros(n_words, dtype=int)
        v[i] = 1
        encoding[word] = v

    # Encode the sonnets.
    encoded_sonnets = []
    for sonnet in sonnets:
        encoded = []
        for i, line in enumerate(sonnet):
            encoded.extend([encoding[word] for word in nltk.word_tokenize(line)])
            encoded.append(encoding['\n'])
        encoded_sonnets.append(np.array(encoded, dtype=int))

    return encoding, encoded_sonnets

def encode_words_word2vec(sonnets, size=100, window=5, iter=100, *args, **kwargs):
    """Given a list of lists, with the outer list containing sonnets and
    the inner list containing lines of sonnets (i.e. the output of
    `utils.load_shakespeare` or `utils.load_spenser`), generate word-based
    Word2Vec encoding of specified size. Internally, each line is tokenized
    using `nltk.word_tokenize`. Any additional arguments are passed to the
    `Word2Vec` constructor.
    
    Preprocessing
    -------------
    - All characters are lowercased.
    - The newline character `\n` is considered a word.

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

    w2v = Word2Vec(sentences, size=size, min_count=1, window=window, iter=iter, *args, **kwargs)
    encoding = w2v.wv

    # Encode.
    encoded_sonnets = []
    for sonnet in sentences:
        encoded_sonnets.append(np.array([encoding[word] for word in sonnet], dtype=int))

    return w2v.wv, encoded_sonnets


def create_sequences_sonnets(sonnets):
    """
    This creates sequences as done in Homework 6, by mapping each word
    to an integer in order to create a series of sequences. This function
    specifically makes entire sonnets into individual sequences
    and returns the list of processed sonnets back to be used in the basic
    HMM notebook for generation.
    """
    sequences = []
    obs_counter = 0
    obs_map = {}
    for sonnet in sonnets:
        sequence = []
        for i, line in enumerate(sonnet):
            split = line.split()
            for word in split:
                word = re.sub(r'[^\w]', '', word).lower()
                if word not in obs_map:
                    # Add unique words to the observations map.
                    obs_map[word] = obs_counter
                    obs_counter += 1
                
                # Add the encoded word.
                sequence.append(obs_map[word])
            
        # Add the encoded sequence.
        sequences.append(sequence)


    return obs_map, sequences


def get_rhymes(sonnets):
    """
    This is a function that takes in all the sonnets and then attempts to 
    store all of the words at the end of the sonnets in order to determine
    which words rhyme with which other words. It then returns a completed
    dictionary of rhyming words back to be used in the reverse seeding
    generation of the sonnets
    """
    
    pass
