import os
import pickle
import re

import nltk

# Paths to text files
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
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
        
def save_pickle(obj, path):
    """Pickle and save a Python object to the given path.
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    
def load_pickle(path):
    """Load a pickled Python object from the given path.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_shakespeare():
    """Load shakespeare.txt. Returns a list of lists.
    The outer list contains sonnets, the inner list contains lines.
    All leading and trailing spaces are removed.

    This function does no preprocessing.
    """
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
    with open(SPENSER_PATH, 'r') as f:
        text = f.read()
    return [
        LINE_PARSER.findall(sonnet)
        for sonnet in SPENSER_PARSER.findall(text)
    ]

def syllable_dic():
    """Load Syllable_dictionary.txt. Returns a dictionary; keys are
    words and values are syllables.    
    """
    f = open(SYLLABLE_PATH, 'r')
    syllable_data = f.read()
    f.close()
    
    syllable_data_split = syllable_data.split('\n')
    syllable_data_split.pop()

    keys = []
    vals = []
    for elem in syllable_data_split:
        elem_split = elem.split(' ')
        keys.append(elem_split[0])
        if len(elem_split) == 2:
            vals.append(elem_split[1])
        elif len(elem_split) > 2:
            if 'E' in elem_split[-1]:
                vals.append(elem_split[1])
            else:
                vals.append(elem_split[-1])
    dic = {keys[i]: int(vals[i]) for i in range(len(keys))}
    
    return dic