#!/usr/bin/env python

import re, cPickle, os, gzip, sys, math, codecs

PENN_TO_WN = ({'JJ':'a','JJR':'a','JJS':'a','CD':'a','RB':'r',
              'RBR':'r','RBS':'r','RP':'r','IN':'r',
              'NN':'n','NNS':'n','NNP':'n','NNPS':'n',
              'VBP':'v','VB':'v','VBD':'v','VBG':'v',
              'VBN':'v','VBZ':'v','VBP':'v','MD':'v'})

def save_pickle(data, path):
    o = gzip.open(path, 'wb')
    cPickle.dump(data, o)
    o.close()

def load_pickle(path):
    i = gzip.open(path, 'rb')
    data = cPickle.load(i)
    i.close()
    return data

def save_text(data, path):
    o = codecs.open(path, encoding='utf-8', mode='w')
    o.write(data)        
    o.close()

def die(msg):
    print '\nERROR: %s' %msg
    sys.exit()

def logit(x, y=1):
    return 1.0 / (1 + math.e ** (-1*y*x))

