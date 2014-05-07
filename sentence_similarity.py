# __author__ = Leila Chan Currie (l.chancurrie@utoronto.ca)

'''Calculates similarity of 2 sentences.'''

import nltk
import string
import subprocess
import util
#import word2vec

import similarity_measures

from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from stanford_tagger import POSTagger

import os
java_path = "/u/leila/jdk1.7.0_55/bin/java"
os.environ['JAVAHOME'] = java_path

PRINT = True

LMTZR = WordNetLemmatizer()
TAGGER = POSTagger('stanford-postagger/models/english-bidirectional-distsim'
                   '.tagger', 'stanford-postagger/stanford-postagger.jar')

# Pronoun tags are PRP, PRP$, WP, WP$
# IN is the tag for prepositions or subordinating conjunctions
# Remaining tags refer to punctuation
STOPWORD_TAGS = ['PRP', 'PRP$', 'WP', 'WP$', 'IN', '#', '$', '"', "(", ")", ",",
                 '.', ':', '``', '\'\'']
PARTS_OF_SPEECH = { wn.NOUN: ['NN', 'NNS', 'NNP', 'NNPS', 'n'],
                    wn.VERB: ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'v'],
                  }


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
      synsets = wn.synsets(test_compound_word)
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

  if PRINT:
    print 'Tokens: %s' % str(matching_tokens)

  return matching_tokens


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

  if PRINT:
    print 'Lemmatized tokens: %s' % str(lemmatized_words)

  return lemmatized_words


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

  total_normalized_score = 0
  total_scaled_score = 0
  for measure in similarity_measures.SIMILARITY_MEASURES:
    score = avg_max_wn_similarity(s1_synsets, s2_synsets, measure)

    if measure in ['path', 'lin', 'wup']: # Scores with range [0,1]
      total_normalized_score += score
    else:
      total_scaled_score += score

    if PRINT:
      print "Score for %s: %f" % (measure, score)

  avg_score = (float(total_normalized_score)/3) # 3 measures with range [0,1]
  avg_scaled_score = (float(total_scaled_score)/3) # 3 scaled measures
  overall_avg = (avg_scaled_score + avg_score)/2

  if PRINT:
    print "Average WordNet score: %f" % overall_avg

  return avg_score

    
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
      avg_noun_wn_score = combined_wn_similarity(anaphor, sentence, wn.NOUN)
      if PRINT:
        print 'VERBS ----------------------'
      avg_verb_wn_score = combined_wn_similarity(anaphor, sentence, wn.VERB)

      #avg_word2vec_score = avg_max_similarity(anaphor, sentence,
        #similarity_measures.word2vec_similarity)
      avg_LSA_score = avg_max_similarity(anaphor, sentence,
                                         similarity_measures.LSA_similarity)

      # Average the different similarity scores to get an overall score for
      # this sentence
      overall_score = (avg_noun_wn_score + avg_verb_wn_score + avg_LSA_score)
                     #  + avg_word2vec_score)
      results[(overall_score)/3] = sentence
      if PRINT:
        print 'LSA score: %f' % avg_LSA_score
        print 'Overall avg score: %f' % (overall_score/3)
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
