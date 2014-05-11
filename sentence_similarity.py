#!/usr/bin/env python

'''Calculates similarity of 2 sentences.'''

__author__ = 'leila@cs.toronto.edu'

import similarity_measures
import subprocess
import util

from nltk.corpus import wordnet as wn
from sentence_to_synset import sentence_to_synset

PRINT = True

def avg_max_similarity(sentence1, sentence2, sim_func):
  '''Returns the average maximum similarity score between the words in sentence1
  and sentence 2.

  Args:
    sentence1: A string.
    sentence2: A string.
    sim_func: The similarity function to be used.
      ie similarity_measures.word2vec_similarity or
      similarity_measures.LSA_similarity.

  Returns:
    A float. The average maximum similarity score between the words in sentence1
  and sentence 2.
  '''
  total_sim_score = 0
  denom = 0
  sentence1 = split_into_words(sentence1)
  sentence2 = split_into_words(sentence2)

  for word1 in sentence1:
    max_score = -1 # Max sim score between word1 and all words in sentence2
    for word2 in sentence2:
      try:
        similarity_score = sim_func(word1, word2)
      except KeyError:
        # For LSA, word was not in dict
        similarity_score = -1

      if similarity_score > max_score:
        max_score = similarity_score
    if max_score >= 0:
      total_sim_score += max_score
      denom += 1

  return 0 if not sentence1 or not denom else float(total_sim_score)/denom


def avg_max_wn_similarity(s1_synsets, s2_synsets, measure):
  '''Returns the average maximum similarity score for the synsets in s1_synsets
  and s2_synsets.

  Args:
    s1_synsets: A list of lists of synsets.
    s2_synsets: A list of lists of synsets.
    measure: A string. The similarity measure to be used. Must be in
      similarity_measures.SIMILARITY_MEASURES.

  Returns:
    The average of the maximum similarity scores between each list of synsets i
    s1_synsets and s2_synsets. A float.
  '''
  total_sim_score = 0
  for synset1 in s1_synsets:
    max_score = 0 # Max sim score between synset1 and all syns in s2_synsets
    for synset2 in s2_synsets:
      for syn1 in synset1:
        for syn2 in synset2:
          if measure == 'lesk':
            similarity_score = similarity_measures.lesk_similarity('lesk', syn1,
                                                                   syn2)
          elif measure == 'vector':
            similarity_score = similarity_measures.lesk_similarity('vector',
                                                                   syn1, syn2)
          else:
            similarity_score = similarity_measures.wn_similarity(syn1, syn2,
                                                                 measure)

          if similarity_score > max_score:
            max_score = similarity_score
    total_sim_score += max_score

  return 0 if not s1_synsets else float(total_sim_score)/len(s1_synsets)

  
def first_sense_wn_similarity(s1_synsets, s2_synsets, measure):
  '''Returns the average of the max similarity scores between the first sense
  of each list of synsets in s1_synsets and s2_synsets.

  Args:
    s1_synsets: A list of lists of synsets.
    s2_synsets: A list of lists of synsets.
    measure: A string. The similarity measure to be used. Must be in
      similarity_measures.SIMILARITY_MEASURES.

  Returns:
    The average of the maximum similarity scores between each list of synsets i
    s1_synsets and s2_synsets. A float.
  '''
  # Get the 1st synset in each list of synsets
  s1_synsets = [synset[0] for synset in s1_synsets]
  s2_synsets = [synset[0] for synset in s2_synsets]

  total_score = 0
  for syn1 in s1_synsets:
    max_score = 0 # Max sim score between syn1 and all the syns in s2_synsets
    for syn2 in s2_synsets:
      similarity_score = similarity_measures.wn_similarity(syn1, syn2, measure)
      if similarity_score > max_score:
        max_score = similarity_score
    total_score += max_score

  return 0 if not s1_synsets else float(total_score)/len(s1_synsets)


def combined_wn_similarity(s1, s2, part_of_speech):
  '''Prints the combined WordNet similarity score for the words in sentences s1
  and s2.

  Args:
    s1: A sentence in string form.
    s2: A sentence in string form.
    part_of_speech: A WordNet part of speech, ie wn.VERB.
      Only words tagged with part_of_speech will be scored for similarity.

  Returns:
    The average score.
  '''
  s1_synsets = sentence_to_synset(s1, part_of_speech)
  s2_synsets = sentence_to_synset(s2, part_of_speech)

  total_score = 0
  for measure in similarity_measures.SIMILARITY_MEASURES:
    score = avg_max_wn_similarity(s1_synsets, s2_synsets, measure)
    total_score += score

    if PRINT:
      print "Score for %s: %f" % (measure, score)

  avg_score = total_score/len(similarity_measures.SIMILARITY_MEASURES)

  if PRINT:
    print "Average WordNet score: %f" % avg_score

  return total_score

    
def compare_sentences(anaphor, sentence_group):
  '''Compares anaphor to all other sentences in sentence_group.

  Args:
    anaphor: A string.
    sentence_group: A dictionary with string (sentence) values.

  Returns:
    A dictionary. The value is a sentence from sentence_group and the key is the
    sentence's overall similarity score to anaphor.
  '''
  results = {}
  for key, sentence in sentence_group.iteritems():
    if key != 'b':
      # This is a sentence to compare to the anaphor sentence
      if PRINT:
        print 'Sentence: %s' % sentence
        print 'NOUNS ----------------------'
      total_noun_wn_score = combined_wn_similarity(anaphor, sentence, wn.NOUN)
      if PRINT:
        print 'VERBS ----------------------'
      total_verb_wn_score = combined_wn_similarity(anaphor, sentence, wn.VERB)

      #avg_word2vec_score = avg_max_similarity(anaphor, sentence,
        #similarity_measures.word2vec_similarity)
      avg_LSA_score = avg_max_similarity(anaphor, sentence,
                                         similarity_measures.LSA_similarity)

      # Average the different similarity scores to get an overall score for
      # this sentence
      overall_score = (total_noun_wn_score + total_verb_wn_score + avg_LSA_score)
                     #  + avg_word2vec_score)
      results[(overall_score)/19] = sentence
      if PRINT:
        print 'LSA score: %f' % avg_LSA_score
        print 'Overall avg score: %f' % (overall_score/19)
        print '----------------------------------------------------------------'

  return results


def get_comparison_results(sentence_group):
  '''Prints the results of comparing the sentence with key 'b' to all others in
  sentence_group.

  Args:
    sentence_group: A dictionary with string (sentence) values.

  Returns:
    None, output is printed.
  '''
  anaphor = sentence_group['b']
  print 'ANAPHOR'
  print anaphor
  results = compare_sentences(anaphor, sentence_group)

  print 'RANKED SENTENCES'
  sorted_results = sorted(results)
  sorted_results.reverse()
  for i, key in enumerate(sorted_results):
    print '%d. %s (%f)' % (i + 1, results[key], key)
  print '----------------------------------------------------------------------'


if __name__ == '__main__':
  
  #print similarity_measures.LSA_similarity('dog', 'cat')
  #print similarity_measures.LSA_similarity('bus', 'banana')


  candidate_source = util.load_pickle('candidate_source.dump')
  for key in candidate_source:
    get_comparison_results(candidate_source[key])
