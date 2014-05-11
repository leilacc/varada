#!/usr/bin/env python

'''Processes a sentence into a list of lists of synsets for each word with a
given part of speech.'''

__author__ = 'leila@cs.toronto.edu'

from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from tag import tag

LMTZR = WordNetLemmatizer()
PARTS_OF_SPEECH = { wn.NOUN: ['NN', 'NNS', 'NNP', 'NNPS', 'n'],
                    wn.VERB: ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'v'],
                  }

def sentence_to_synset(sentence, part_of_speech):
  '''Processes the string sentence into a list of lists of synsets of its words
  that match part_of_speech.

  Args:
    sentence: A sentence in string form.
    part_of_speech: A WordNet part of speech, ie wn.VERB.
      Only words tagged with part_of_speech will be scored for similarity.
      TODO: None if comparing for all parts of speech?

  Returns:
    A list of lists. Each list contains all synsets for a word in tokens whose
    part of speech matches pos.
  '''
  tokens = get_pos(sentence, PARTS_OF_SPEECH[part_of_speech])
  tokens = lemmatize(tokens, part_of_speech)
  synsets = get_synsets(tokens, part_of_speech)
  return synsets


def get_pos(sentence, parts_of_speech):
  '''Returns the tokens in sentence that have a match in parts_of_speech.

  Args:
    sentence: A sentence in string form.
    parts_of_speech: A list of parts of speech in string form, ie 'NN' or 'VB'

  Returns:
    A list of tokens in tagged_sentence whose part of speech has a match in
    parts_of_speech.
  '''
  matching_tokens = []
  tagged_sentence = tag(sentence)

  for tagged_pair in tagged_sentence:
    cur_token = tagged_pair[0]
    cur_pos = tagged_pair[1]
    if cur_pos in parts_of_speech:
      matching_tokens.append(cur_token)

  #if PRINT:
  #  print 'Tokens: %s' % str(matching_tokens)

  return matching_tokens


def lemmatize(tokens, part_of_speech):
  '''Returns the lemmatized words in tokens.

  Args:
    tokens: A list of words
    part_of_speech: A WordNet part of speech, ie wn.VERB.
      Only words tagged with part_of_speech will be scored for similarity.
      TODO: None if comparing for all parts of speech?

  Returns:
    A list of the lemmatized versions of the words in tokens.
  '''
  # Note: The wordnet lemmatizer only knows four parts of speech:
  # {'a': 'adj', 'n': 'noun', 'r': 'adv', 'v': 'verb'}
  lemmatized_words = []
  for word in tokens:
    if part_of_speech == wn.VERB:
      lmtzd_word = LMTZR.lemmatize(word, 'v')
    elif part_of_speech == wn.NOUN:
      lmtzd_word = LMTZR.lemmatize(word, 'n')
    lemmatized_words.append(lmtzd_word)

  #if PRINT:
    #print 'Lemmatized tokens: %s' % str(lemmatized_words)

  return lemmatized_words


def get_synsets(tokens, pos):
  '''Returns all non-empty WordNet synsets with part of speech pos for each word
  in tokens.

  Args:
    tokens: A list of words.
    pos: The part of speech of the words. ie wn.VERB, wn.NOUN

  Returns:
    A list of lists. Each list contains all synsets for a word in tokens whose
    part of speech matches pos.
  '''
  synsets = []
  for word in tokens:
    synset = wn.synsets(word, pos=pos)
    if synset:
      synsets.append(synset)
  return synsets
