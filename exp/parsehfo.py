#!/usr/bin/env python

def parsehfo(filename):
  f = open(filename, 'r')
  g = open('/dev/stdout', 'w')
  g.write('goals,trials,frames\n')
  for line in f.readlines():
    if line.find("EndOfTrial") != 0:
      continue
    try:
      tokens = line.split(' ')
      cum_goals = int(tokens[1])
      cum_trials = int(tokens[3])
      cum_frames = int(tokens[4])
      g.write('%d,%d,%d\n' % (cum_goals, cum_trials, cum_frames))
    except Exception:
      continue
  f.close()
  g.close()

if __name__ == '__main__':
  import sys
  parsehfo(sys.argv[1])

