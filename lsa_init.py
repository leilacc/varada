#!/usr/bin/env python

'''Initializes an LSA model that can be used for word-word similarity queries'''

__author__ = 'leila@cs.toronto.edu'

import gensim

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Number of dimensions used for LSA
NUM_DIMENSIONS = 100

def generate_LSA_models(file_prefix):
  '''Returns a gensim LSA index, corpus, and dictionary from the files that
  share file_prefix.

  Args:
    file_prefix: The file prefix for _bow.mm and _wordids.txt files.

  Returns: 
    A gensim.corpora.Dictionary, gensim.similarities.MatrixSimilarity, and
    gensim.corpora.MmCorpus object that can be used for similarity queries.
  '''
  corpus_filename = file_prefix + '_bow.mm'
  dict_filename = file_prefix + '_wordids.txt'

  corpus = gensim.corpora.MmCorpus(corpus_filename)
  dictionary = gensim.corpora.Dictionary().load_from_text(dict_filename)
  
  tfidf = gensim.models.TfidfModel(corpus)
  corpus_tfidf = tfidf[corpus]
  lsi = gensim.models.LsiModel(corpus_tfidf, id2word=dictionary,
                               num_topics=NUM_DIMENSIONS)

  termcorpus = gensim.matutils.Dense2Corpus(lsi.projection.u.T)
  index = gensim.similarities.MatrixSimilarity(termcorpus) 

  return (dictionary, index, termcorpus)
