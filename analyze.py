'''Analyze results'''

import operator
import os
import csv
import numpy as np
import util
from scipy import stats

total_ante_score = []
total_cand_score = []
total_ante_first = 0
total_cand_first = 0
total_ante_last = 0
total_cand_last = 0
antecedents = []

sim = -1 # Index of similarity score to use

candidate_source = util.load_pickle('candidate_source.dump')
crowd_results = util.load_pickle('crowd_results.dump')

for dirpath, dirnames, filenames in os.walk('results'):
  if dirpath.split('/')[-1] == 'results':
    continue

  print dirpath.split('/')[-1]
  ante_score = []
  cand_score = []
  ante_first = 0
  cand_first = 0
  ante_last = 0
  cand_last = 0

  for filename in filenames:
    lines = []
    with open(os.path.join(dirpath, filename), 'r') as f:
      reader = csv.reader(f)
      for row in reader:
        lines.append(row)

    lines.pop(0) # Get rid of header
    # Convert to floats
    lines = [[line[0]] + map(float, line[1:]) for line in lines]
    lines.sort(key=operator.itemgetter(sim)) # Sort by value of sim measure

    for i, line in enumerate(lines):
      if line[0] == 'Antecedent':
        ante_score.append(line[sim])
        key = filename.split('.')[0]

        antecedents.append([key, line[sim]])
        if i == 0:
          ante_last += 1
        elif i == len(lines) - 1:
          ante_first += 1
          print '---------------------------------------------------------'
          print 'Anaphor: %s' % (candidate_source[key]['b'])
          print '-----'
          print 'Antedecent %f\n%s' % (line[-1], crowd_results[key][2])
          for l in lines:
            if l[0] != 'Antecedent':
              print '-----'
              print 'Candidate: %f\n%s' % (l[-1], candidate_source[key][l[0]])
      else:
        cand_score.append(line[sim])
        if i == 0:
          cand_last += 1
        elif i == len(lines) - 1:
          cand_first += 1

  mean_a_f = np.mean(ante_first)
  mean_c_f = np.mean(cand_first)
  p_a_f = mean_a_f / (mean_a_f + mean_c_f)
  p_c_f = mean_c_f / (mean_a_f + mean_c_f)
  mean_a_l = np.mean(ante_last)
  mean_c_l = np.mean(cand_last)
  p_a_l = mean_a_l / (mean_a_l + mean_c_l)
  p_c_l = mean_c_l / (mean_a_l + mean_c_l)
  print 'Ante mean: %f' % np.mean(ante_score)
  print 'Candidate mean: %f' % np.mean(cand_score)
  print 'Ante first: %d (%f)' % (mean_a_f, p_a_f)
  print 'Candidate first: %d (%f)' % (mean_c_f, p_c_f)
  print 'Ante last: %d (%f)' % (mean_a_l, p_a_l)
  print 'Candidate last: %d (%f)' % (mean_c_l, p_c_l)
  print '--------------------------------------------------------------'

  total_ante_score.extend(ante_score)
  total_cand_score.extend(cand_score)
  total_ante_first += ante_first
  total_ante_last += ante_last
  total_cand_first += cand_first
  total_cand_last += cand_last

mean_a_f = np.mean(total_ante_first)
mean_c_f = np.mean(total_cand_first)
p_a_f = mean_a_f / (mean_a_f + mean_c_f)
p_c_f = mean_c_f / (mean_a_f + mean_c_f)
mean_a_l = np.mean(total_ante_last)
mean_c_l = np.mean(total_cand_last)
p_a_l = mean_a_l / (mean_a_l + mean_c_l)
p_c_l = mean_c_l / (mean_a_l + mean_c_l)
print 'Total ante mean: %f' % np.mean(total_ante_score)
print 'Total candidate mean: %f' % np.mean(total_cand_score)
print 'Total ante first: %d (%f)' % (mean_a_f, p_a_f)
print 'Total candidate first: %d (%f)' % (mean_c_f, p_c_f)
print 'Total ante last: %d (%f)' % (mean_a_l, p_a_l)
print 'Total candidate last: %d (%f)' % (mean_c_l, p_c_l)
print 'ttest ante-cand scores: %f, %f' % stats.ttest_ind(total_ante_score, total_cand_score)

num_a1 = 0
for key in crowd_results:
  if crowd_results[key][1] == 'a1':
      num_a1 += 1
print num_a1

antecedents.sort(key=operator.itemgetter(-1))
print antecedents
