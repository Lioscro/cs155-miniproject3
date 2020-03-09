import os
import re

import nltk

# Paths to text files
ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
SHAKESPEARE_PATH = os.path.join(DATA_DIR, 'shakespeare.txt')
SPENSER_PATH = os.path.join(DATA_DIR, 'spenser.txt')
SYLLABLE_PATH = os.path.join(DATA_DIR, 'Syllable_dictionary.txt')

SHAKESPEARE_PARSER = re.compile(r'\s{19}[0-9]+\n(?P<sonnet>.+?)(?:\n\n\n|$)', re.DOTALL)
SPENSER_PARSER = re.compile(r'[IVXL]+\n\n(?P<sonnet>.+?)(?:\n\n|$)', re.DOTALL)
LINE_PARSER = re.compile(r'\s*(?P<line>.+)\s*')

def check_nltk_package(package):
    try:
        nltk.data.find(package)
    except LookupError:
        nltk.download(package)

def load_shakespeare():
    """Load shakespeare.txt. Returns a list of lists.
    The outer list contains sonnets, the inner list contains lines.
    All leading and trailing spaces are removed.

    This function does no preprocessing.
    """
    check_nltk_package('tokenizers/punkt')

    with open(SHAKESPEARE_PATH, 'r') as f:
        text = f.read()
    return [
        LINE_PARSER.findall(sonnet)
        for sonnet in SHAKESPEARE_PARSER.findall(text)
    ]

def load_spenser():
    """Load shakespeare.txt. Returns a list of lists.
    The outer list contains sonnets, the inner list contains lines.
    All leading and trailing spaces are removed.

    This function does no preprocessing.
    """
    check_nltk_package('tokenizers/punkt')

    with open(SPENSER_PATH, 'r') as f:
        text = f.read()
    return [
        LINE_PARSER.findall(sonnet)
        for sonnet in SPENSER_PARSER.findall(text)
    ]
