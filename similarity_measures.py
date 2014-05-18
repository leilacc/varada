#!/usr/bin/env python

'''Various methods for calculating word/synset similarity.'''

__author__ = 'leila@cs.toronto.edu'

import json
import math
import urllib
import urllib2
import word2vec

from nltk.corpus import wordnet_ic # For corpuses

import lsa_init

WORD2VEC_MODEL = word2vec.Word2Vec.load_word2vec_format(
  '/p/cl/varada/word2vec-GoogleNews-vectors-negative300.bin', binary=True)

# Models for LSA similarity
#WIKI_DICT, WIKI_INDEX, WIKI_CORPUS = lsa_init.generate_LSA_models(
#  '/u/leila/gensim_wikicorpus/articles27')

# Initialize corpuses
BROWN_IC = wordnet_ic.ic('ic-brown.dat')
SEMCOR_IS = wordnet_ic.ic('ic-semcor.dat')
# Set corpus
CORPUS = BROWN_IC
if CORPUS == BROWN_IC:
  CORPUS_SIZE = 1014312
else:
  CORPUS_SIZE = 2

SIMILARITY_MEASURES = ['path', 'lch', 'wup', 'res', 'jcn', 'lin',
                       'lesk', 'vector']
# Max lch score is 3.6889
SCALED_MEASURES = {'lch': 1/3.6889, 'jcn': 1, 'res': 1/math.log(CORPUS_SIZE, 2)}


def LSA_similarity(word1, word2): 
  '''Returns the LSA similarity score of word1 and word2.
  
  Args:
    word1: A string.
    word2: A string.

  Returns:
    A float representing the similarity of word1 and word2 as calculated by
    the LSA WIKI_* models, normalized to the [0, 1] range.

  Throws:
    KeyError if either word1 or word2 is not in WIKI_DICT
  '''
  w1_id = WIKI_DICT.token2id[word1]
  w2_id = WIKI_DICT.token2id[word2]

  word1_vec = list(WIKI_CORPUS)[w1_id]
  sims = WIKI_INDEX[word1_vec] 

  # Cosine range is [-1, 1]. Normalize to [0, 1]
  return (sims[w2_id] + 1)/2.0


def word2vec_similarity(word1, word2):
  '''Returns the word2vec similarity score of word1 and word2.
  
  Args:
    word1: A string.
    word2: A string.

  Returns:
    A float representing the similarity of word1 and word2 as calculated by
    the word2vec model WORD2VEC_MODEL.
  '''
  return WORD2VEC_MODEL.similarity(word1, word2)


def wn_similarity(synset1, synset2, measure):
  '''Returns a score denoting how similar 2 word senses are, based on measure.

  Args:
    synset1: A WordNet synset, ie wn.synset('dog')
    synset2: A WordNet synset to be compared to synset1
    measure: A string. The similarity measure to be used. Must be in
      SIMILARITY_MEASURES.

  Returns:
    A score denoting how similar synset1 is to synset2 based on measure.
  '''
  ic_measures = ['res', 'jcn', 'lin'] # These measures require a corpus
  similarity_function = measure + '_similarity'

  if measure in SCALED_MEASURES:
    # Score must be normalized
    scale = SCALED_MEASURES[measure]
  else:
    scale = 1

  if measure in ic_measures:
    # Equivalent to calling synset1.$sim_fn(synset2, corpus)
    return min(1, scale*getattr(synset1, similarity_function)(synset2, CORPUS))
  else:
    # Equivalent to calling synset1.$sim_fn(synset2)
    return min(1, scale*getattr(synset1, similarity_function)(synset2))


#TODO: change name, check for valid return values
def lesk_similarity(sim_type, synset1, synset2):
  '''Returns a score denoting how similar 2 word senses are based on Adapted
  Lesk.

  Args:
    sim_type: A string, either "lesk" or "vector".
    synset1: A WordNet Synset, ie wn.synset('dog')
    synset2: A WordNet Synset to be compared to synset1

  Returns:
    A score denoting how similar synset1 is to synset2 based on the Adapted
    Lesk algorithm.
  '''
  synset1 = synset1.name.replace('.', '#')
  synset2 = synset2.name.replace('.', '#')

  data = json.dumps({"type" : sim_type, "syn1" : synset1, "syn2": synset2})
  headers = {
        "User-Agent" : "pyclient",
  }

  request = urllib2.Request("http://127.0.0.1:8080/api", data, headers)
  try:
    response = urllib2.urlopen(request)
    score = response.read().split(':')[1][0:-1]
    return min(float(score), 1.0) # Sometimes lesk scores are slightly > 1
  except urllib2.URLError:
    raise urllib2.URLError('Perl server for Lesk similarity refused connection')
  except ValueError:
    # Score is an empty string because algorithm failed to return a score
    # Known to occur for any comparisons involving shift_key#n#01 
    return 0
