import nltk
nltk.download('cmudict')
from nltk.corpus import cmudict
from nltk.tokenize import word_tokenize, wordpunct_tokenize, sent_tokenize

def cmp(p, wt):
  pt = set([s.pos for s in wordnet.synsets(wt)])
  return len(pt & p) > 0 or len(p) == 0 or len(pt) == 0
 
def rhymes(s):
  try:
    (w, l, p) = s[0]
    try:
      pos = set([s.pos for s in wordnet.synsets(w)])
      filtered = [wt for (wt, pt) in cmudict.entries() 
                  if l == len(pt) 
                  and p[-2:] == pt[-2:] 
                  and (nltk.distance.edit_distance(w, wt) > 2 \
                  or not w[0:2] == wt[0:2])
                  and cmp(pos, wt)
                  and len(nltk.corpus.wordnet.synsets(wt)) > 0]
      return filtered
    except:
      return [w]
  except:
    return []


def rhyme_scheme(sonnets):
    tokens = []
    for sonnet in sonnets:
        to_add = [wordpunct_tokenize(s) for s in sonnet]
        tokens += to_add

    punct = set(['.', ',', '!', ':', ';'])
    filtered = [ [w for w in sentence if w not in punct ] for sentence in tokens]
    last = [ sentence[len(sentence) - 1] for sentence in filtered]

    syllables = \
        [[(w, len(p), p) for (w, p) in cmudict.entries() if word == w] \
            for word in last]

    return [rhymes(s) for s in syllables]

def get_individual_rhymes(sonnets):
    all_rhymes = []
    for sonnet in sonnets:
        tokens = [wordpunct_tokenize(s) for s in sonnet]
        punct = set(['.', ',', '!', ':', ';'])
        filtered = [ [w for w in sentence if w not in punct ] for sentence in tokens]
        last = [ sentence[len(sentence) - 1] for sentence in filtered]

        # now that we have a list of the last words, check the sonnets
        # specifically if it is the ababcdcdefefgg or the other scheme
        pairs = [[last[0], last[2]], [last[1], last[3]], \
                    [last[4], last[6]], [last[5], last[7]], \
                    [last[8], last[10]], [last[9], last[11]], \
                    [last[12], last[13]]]
        all_rhymes += pairs

    return all_rhymes











