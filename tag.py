#!/usr/bin/env python

'''Tags each word in a sentence with its part of speech. Combines compound words
and removes stop words.'''

__author__ = 'leila@cs.toronto.edu'

import nltk
import string
import stanford_tagger

# Must set JAVAHOME for the stanford tagger
import os
java_path = "/u/leila/jdk1.7.0_55/bin/java"
os.environ['JAVAHOME'] = java_path

TAGGER = stanford_tagger.POSTagger('stanford-postagger/models/'
                                   'english-bidirectional-distsim.tagger',
                                   'stanford-postagger/stanford-postagger.jar')

# Pronoun tags are PRP, PRP$, WP, WP$
# IN is the tag for prepositions or subordinating conjunctions
# Remaining tags refer to punctuation
STOPWORD_TAGS = ['PRP', 'PRP$', 'WP', 'WP$', 'IN', '#', '$', '"', "(", ")", ",",
                 '.', ':', '``', '\'\'']

def tag(sentence):
  '''Tags a sentence with its parts of speech. Combines compound words and
  removes stop words.

  Args:
    sentence: A string. The sentence whose parts of speech will be tagged.

  Returns:
    The same sentence, but as a list of tuples with its parts of speech tagged.
    Compound words are tagged as one and stop words are removed.
    eg [('hello', 'UH'), ('how', 'WRB'), ('are', 'VBP'), ('you', 'PRP')]
  '''
  tagged_sentence = TAGGER.tag(nltk.word_tokenize(sentence))
  #  print "Original tagged sentence:\n%s" % tagged_sentence
  tagged_sentence = combine_compound_words(tagged_sentence, sentence)
  return remove_stopwords(tagged_sentence)


def combine_compound_words(tagged_sentence, sentence):
  '''Replaces individual words that make up a compound word in tagged_sentence
  with the actual compound word.

  Args:
    tagged_sentence: A list of tuples of strings. Each tuple is of form
      (word, tag)
  
  Returns:
    The same tagged_sentence, but with constituent words of compound words
    combined into a single compound word.
  '''
  compound_words = find_compound_words(sentence)
  for c_word in compound_words:
    i = c_word[0]
    j = c_word[1]
    synset = c_word[2]
    constituent_words = ' '.join(split_into_words(sentence)[i:j + 1])
    (tagged_sentence, k) = remove_individual_compound_words(constituent_words,
                                                            tagged_sentence)
    tagged_sentence.insert(k, (constituent_words, synset.pos))

  return tagged_sentence


def find_compound_words(sentence):
  '''Identifies compound words in sentence. Takes the longest possible compound
  word, if one is the subset of another.
  
  Args:
    sentence: A string.
    
  Returns:
    A list of tuples of each compound word's start and end indices and synset
    word, for all compound words found in sentence.
    eg [(0, 1, Synset)] if a 2-word compound word starts the sentence.
  '''
  all_compound_words = []
  sentence = split_into_words(sentence)
  
  # For each word, iteratively add on the words that follow to see if together
  # they form a compound word with a WordNet synset.
  for i, word in enumerate(sentence):
    test_compound_word = word
    for j in range(i + 1, len(sentence)):
      test_compound_word += ('_' + sentence[j])
      synsets = nltk.corpus.wordnet.synsets(test_compound_word)
      if synsets:
        for synset in synsets:
          all_compound_words.append((i, j, synset, test_compound_word))
  
  # Remove compound words that are a subset of another word
  final_compound_words = []
  for i in range(len(all_compound_words)):
    i1 = all_compound_words[i][0]
    j1 = all_compound_words[i][1]
    w1 = all_compound_words[i][3]
    subset = False
    for j in range(len(all_compound_words)):
      i2 = all_compound_words[j][0]
      j2 = all_compound_words[j][1]
      w2 = all_compound_words[j][3]
      if i != j and w1 in w2 and i1 >= i2 and j1 <= j2:
        subset = True
    if not subset:
      final_compound_words.append(all_compound_words[i])
  return final_compound_words


def split_into_words(sentence):
  '''Splits sentence into a list of words without punctuation.

  Args:
    sentence: A string

  Returns:
    A list of the words in sentence, with all punctuation removed.
  '''
  return "".join([i for i in sentence if i not in string.punctuation]).split()


def remove_individual_compound_words(constituent_words, tagged_sentence):
  '''Removes the individual words in tagged_sentence that are constituent words
  of a compound word.

  Args:
    constituent_words: A string of words that together form a compound word.
    tagged_sentence: A list of tuples of strings. Each tuple is of form
      (word, tag)

  Returns:
    A tuple. The first element is the tagged_sentence with the constituent
    words removed, and the second element is the index in tagged_sentence
    where the removal began.
  '''
  tagged_constituents = TAGGER.tag(nltk.word_tokenize(constituent_words))
  cur_i = 0
  start_i = False
  end_i = 0
  for i, tup in enumerate(tagged_sentence):
    if tup == tagged_constituents[cur_i]:
      cur_i += 1
      if not start_i:
        start_i = i
    elif tup == tagged_constituents[0]:
      start_i = i
      cur_i = 1
    else:
      cur_i = 0

    if cur_i == len(tagged_constituents):
      end_i = i + 1
      break

  return (tagged_sentence[0:start_i] + tagged_sentence[end_i:], start_i)


def remove_stopwords(tagged_sentence):
  '''Removes stopwords from a part-of-speech tagged_sentence.
  Pronouns, prepositions, and punctuation are considered stopwords.

  Args:
    tagged_sentence: A list of tuples of strings. Each tuple is of form
      (word, tag)

  Returns:
    The list, but without any tuples containing stop words.
  '''
  sentence_without_stopwords = []
  for word in tagged_sentence:
    tag = word[1]
    if tag not in STOPWORD_TAGS:
      sentence_without_stopwords.append(word)

  return sentence_without_stopwords
