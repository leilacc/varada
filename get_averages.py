import os
import csv
import numpy as np

for dirpath, dirnames, filenames in os.walk('/u/leila/sensim/results'):
  for filename in filenames:
    lines = []
    with open(os.path.join(dirpath, filename), 'r') as f:
      reader = csv.reader(f)
      for row in reader:
        lines.append(row)

    header = [lines.pop(0)]
    header[0].append('Average')
    lines = [line + [np.mean(map(float, line[1:]))] for line in lines]
    header.extend(lines)

    with open(os.path.join(dirpath, filename), 'w') as f:
      writer = csv.writer(f, delimiter=',')
      writer.writerows(header)
    
