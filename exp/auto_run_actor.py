#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os

def main(start_cmd, res_dir_prefix):
  # Run all methods
  methods = [-1,0,1,9,11]
  repeat = 3
  for method in methods:
    for rep in range(repeat):
      res_path = res_dir_prefix + '/method%d.rep%d.log' % (method, rep)
      os.system(start_cmd)
      os.system('sleep 600')
      os.system('./kill.sh')
      os.system('mv test.log %s' % res_path)
      # Now cmd is finished
  os.system('mv run00.log %s' % res_dir_prefix + '/base.log')

if __name__ == '__main__':
  main('./start_il.sh 0.1', 'withp_e0.1')
  main('./start_il.sh 0.2', 'withp_e0.2')
  main('./start_il.sh 0.3', 'withp_e0.3')
  main('./start_il.sh 0.4', 'withp_e0.4')
  main('./start_il.sh 0.5', 'withp_e0.5')

