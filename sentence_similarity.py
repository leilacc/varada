# Author: Leila Chan Currie (l.chancurrie@utoronto.ca)

'''Calculates similarity of 2 sentences.'''


import nltk
import util
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag.stanford import POSTagger

# Initialize corpuses
BROWN_IC = wordnet_ic.ic('ic-brown.dat')
SEMCOR_IS = wordnet_ic.ic('ic-semcor.dat')

LMTZR = WordNetLemmatizer()
TAGGER = POSTagger('/Users/Leila/Downloads/stanford-postagger/models/'
                   'english-bidirectional-distsim.tagger', '/Users/Leila/'
                   'Downloads/stanford-postagger/stanford-postagger.jar')

SIMILARITY_MEASURES = ['path', 'lch', 'wup', 'res', 'jcn', 'lin']

PARTS_OF_SPEECH = { wn.NOUN: ['NN', 'NNS', 'NNP', 'NNPS'],
                    wn.VERB: ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
                  }


def similarity(synset1, synset2, measure):
  '''Returns a score denoting how similar 2 word senses are, based on measure.

  Args:
    synset1: A WordNet synset, ie wn.synset('dog')
    synset2: A WordNet synset to be compared to synset1
    measure: A string. The similarity measure to be used. Must be in
      SIMILARITY_MEASURES.

  Returns:
    A score denoting how similar synset1 is to synset2 based on measure.
  '''
  ic_measures = ['res', 'jcn', 'lin'] # These measures require a 2nd arg
  similarity_function = measure + '_similarity'

  if measure in ic_measures:
    # Equivalent to calling synset1.$sim_fn(synset2, corpus)
    return getattr(synset1, similarity_function)(synset2, BROWN_IC)
  else:
    # Equivalent to calling synset1.$sim_fn(synset2)
    return getattr(synset1, similarity_function)(synset2)

def tag(sentence):
  '''Tags a sentence with its parts of speech.

  Args:
    sentence: A string. The sentence whose parts of speech will be tagged.

  Returns:
    The same sentence, but as a list of tuples with its parts of speech tagged.
    eg [('hello', 'UH'), ('how', 'WRB'), ('are', 'VBP'), ('you', 'PRP')]
  '''
  return TAGGER.tag(nltk.word_tokenize(sentence))

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

def avg_max_similarity(s1_synsets, s2_synsets, measure):
  '''Returns the average maximum similarity score for the synsets in s1_synsets
  and s2_synsets.

  Args:
    s1_synsets: A list of lists of synsets.
    s2_synsets: A list of lists of synsets.
    measure: A string. The similarity measure to be used. Must be in
      SIMILARITY_MEASURES.

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
          similarity_score = similarity(syn1, syn2, measure)

          if similarity_score > max_score:
            max_score = similarity_score
    total_sim_score += max_score

  return 0 if not s1_synsets else float(total_sim_score)/len(s1_synsets)
  
def first_sense_similarity(s1_synsets, s2_synsets, measure):
  '''Returns the similarity score between the first senses of the synsets in
  s1_synsets and s2_synsets.

  Args:
    s1_synsets: A list of lists of synsets.
    s2_synsets: A list of lists of synsets.
    measure: A string. The similarity measure to be used. Must be in
      SIMILARITY_MEASURES.

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
      similarity_score = similarity(syn1, syn2, measure)
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

  print 'Lemmatized tokens: %s' % str(lemmatized_words)
  return lemmatized_words

def process_sentence(sentence, part_of_speech):
  '''Processes the string sentence into a list of synsets of its words that
  match part_of_speech.

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

def wordnet_similarity(s1, s2, part_of_speech, avg_max=True):
  '''Prints the combined WordNet similarity score for the words in sentences s1
  and s2.

  Args:
    s1: A sentence in string form.
    s2: A sentence in string form.
    part_of_speech: A WordNet part of speech, ie wn.VERB.
      Only words tagged with part_of_speech will be scored for similarity.
      TODO: None if comparing for all parts of speech?
    avg_max: If True, will print the average maximum similarity score.
      Otherwise, will print the first sense similarity score. 
  '''
  s1_synsets = process_sentence(s1, part_of_speech)
  s2_synsets = process_sentence(s2, part_of_speech)

  for measure in SIMILARITY_MEASURES:
    if avg_max:
      score = avg_max_similarity(s1_synsets, s2_synsets, measure)
    else:
      score = first_sense_similarity(s1_synsets, s2_synsets, measure)
    print "Score for %s: %f" % (measure, score)
    
def compare_sentences(sentence_group):
  '''Compares the sentence with key 'b' to all others in sentence_group.

  Args:
    sentence_group: A dictionary with string (sentence) values.

  Returns:
    None, output is printed.
  '''
  anaphor = sentence_group['b']
  for key, sentence in sentence_group.iteritems():
    if key != 'b':
      # This is a sentence to compare to the anaphor sentence
      print 'Anaphor: %s' % anaphor
      print 'Antecedent: %s' % sentence
      print 'NOUNS ----------------------'
      wordnet_similarity(anaphor, sentence, wn.NOUN)
      print 'VERBS ----------------------'
      wordnet_similarity(anaphor, sentence, wn.VERB)
      print '------------------------------------------------------------------'

if __name__ == '__main__':
#  print similarity(wn.synset('dog.n.01'), wn.synset('cat.n.01'), 'path')
#  print similarity(wn.synset('dog.n.01'), wn.synset('cat.n.01'), 'wup')
  
  #print wordnet_similarity('tuna salad how are you', 'Hi are you ok?', wn.VERB)
  #print '***'
  #print wordnet_similarity('Hi are you ok?', 'tuna salad how are you', wn.VERB)
  #print '***'
  #print wordnet_similarity('greetings how are you', 'Hi are you ok?', wn.VERB)

  candidate_source = util.load_pickle('candidate_source.dump')
  compare_sentences(candidate_source['1'])

