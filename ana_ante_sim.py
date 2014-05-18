#!/usr/bin/env python

'''Calculates similarity between anaphors and actual/candidate antecedents.'''

__author__ = 'leila@cs.toronto.edu'

import os.path
import sentence_similarity
import util
import csv

PRINT = True


def get_comparison_results(sentence_group, actual_antecedent_key, group_key):
  '''Prints the results of comparing the sentence with key 'b' to all others in
  sentence_group.

  Args:
    sentence_group: A dictionary with string (sentence) values.
    actual_antecedent_key: The key to the actual antecedent in sentence_group.
    group_key: The key for this anaphor-candidate grouping. String.

  Returns:
    None.
  '''
  anaphor = sentence_group['b']
  ana_category = get_anaphor_category(anaphor)

  csvheader = ['Similarity Measures', 'noun_path', 'noun_lch', 'noun_wup',
               'noun_res', 'noun_jcn', 'noun_lin', 'noun_lesk', 'noun_vector', 
               'verb_path', 'verb_lch', 'verb_wup', 'verb_res', 'verb_jcn',
               'verb_lin', 'verb_lesk', 'verb_vector', 'word2vec']
  filename = 'results/%s/%s.csv' % (ana_category, group_key) 
  if os.path.isfile(filename):
    # already got these results
    return

  f = open(filename, 'w')
  wr = csv.writer(f, quoting=csv.QUOTE_ALL)
  wr.writerow(csvheader)

  for key, candidate in sentence_group.iteritems():
    if key != 'b':
      name = 'Antecedent' if key == actual_antecedent_key else key
      results = sentence_similarity.compare_sentences(anaphor, candidate)
      wr.writerow([name] + results)

  if PRINT:
    print 'ANAPHOR'
    print anaphor
    print '--------------------------------------------------------------------'

  f.close()


def get_anaphor_category(anaphor):
  '''Determines which shell noun the anaphor uses.

  Args:
    anaphor: The anaphor. String.

  Returns:
    Either 'reason', 'issue', 'decision', 'question', 'possibility', or 'fact'.
  '''
  if 'possibility' in anaphor:
    return 'possibility'
  elif 'issue' in anaphor:
    return 'issue'
  elif 'fact' in anaphor:
    return 'fact'
  elif 'reason' in anaphor:
    return 'reason'
  elif 'decision' in anaphor:
    return 'decision'
  elif 'question' in anaphor:
    return 'question'
  else:
    return 'other'


if __name__ == '__main__':
  
  mod = 0

  candidate_source = util.load_pickle('candidate_source.dump')
  crowd_results = util.load_pickle('crowd_results.dump')
  for key in candidate_source:
    if key.isdigit():
      if int(key) % 3 == mod: 
        try:
          get_comparison_results(candidate_source[key], crowd_results[key][1],
                                 key)
        except KeyError:
          # crowd_results has fewer identifiers than candidate_source because
          # annotators did not agree on the antecedents for some ids
          pass
    elif mod == 0: # some keys aren't ints ie G30
      try:
        get_comparison_results(candidate_source[key], crowd_results[key][1],
                               key)
      except KeyError:
        # crowd_results has fewer identifiers than candidate_source because
        # annotators did not agree on the antecedents for some ids
        pass

