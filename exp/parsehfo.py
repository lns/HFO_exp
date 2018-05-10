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

def count_final_performance(filename):
  f = open(filename, 'r')
  goals = []
  trials= []
  for line in f.readlines():
    if line.find("EndOfTrial") != 0:
      continue
    try:
      tokens = line.split(' ')
      cum_goals = int(tokens[1])
      cum_trials = int(tokens[3])
      cum_frames = int(tokens[4])
      goals.append(cum_goals)
      trials.append(cum_trials)
    except Exception:
      continue
  # count scoring rate after 3000 episodes
  assert len(goals) > 3500
  final_goals  = goals[-1] - goals[3000]
  final_trials = trials[-1] - trials[3000]
  return (float)(final_goals)/(float)(final_trials)

def summary_performance(prefix="withp"):
  epsilons = ['0.1','0.2','0.3','0.4','0.5']
  methods = [-1,0,1,2,10,11,12]
  repeats = [0,1,2]
  g = open('/dev/stdout', 'w')
  g.write('epsilon,method,rep,rate\n')
  for epsilon in epsilons:
    logfile = prefix+"_e%s/base.log" % epsilon
    perf = count_final_performance(logfile)
    g.write('%s,%d,%d,%f\n' % (epsilon, -2, 0, perf)) # Base
    for method in methods:
      for rep in repeats:
        logfile = prefix+"_e%s/method%d.rep%d.log" % (epsilon, method, rep)
        perf = count_final_performance(logfile)
        g.write('%s,%d,%d,%f\n' % (epsilon, method, rep, perf))
  g.close()

if __name__ == '__main__':
  import sys
  #parsehfo(sys.argv[1])
  #print(count_final_performance(sys.argv[1]))
  summary_performance("fitp")

