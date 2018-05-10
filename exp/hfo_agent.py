#!/usr/bin/env python
# encoding: utf-8

import latte
import numpy as np
from random import random, randint
from threading import Thread
from memoire import ReplayMemoryClient
import math, time

class RuleAgent(object):
  def __init__(self):
    pass

  def reset(self):
    pass

  def act(self, obs):
    dir_ball = list(obs[51:53])
    dir_goal = list(obs[13:15])
    prm = dir_ball + dir_ball + dir_goal
    if obs[12]>0:
      action = [2] + prm # kick
    elif dir_ball[1]<0.9:
      action = [1] + prm # turn
    else:
      action = [0] + prm # dash
    v = np.array((1,), dtype=np.float32)
    v[0] = 0.0
    return (action, 0.0, v) # TODO

class RandomAgent(object):
  def __init__(self):
    pass

  def reset(self):
    pass

  def act(self, obs):
    kickable = obs[12]
    if kickable:
      dis_act = randint(0,2)
    else:
      dis_act = randint(0,1)
    theta = 2*3.1416*(random() - 0.5)
    r = random()
    cnt_act = [r*math.cos(theta), r*math.sin(theta)]
    action = [dis_act] + [0]*6
    action[1+2*dis_act+0] = cnt_act[0]
    action[1+2*dis_act+1] = cnt_act[1]
    v = np.array((1,), dtype=np.float32)
    v[0] = 0.0
    return (action, 0.0, v) # TODO

class RoboAgent(object):
  def __init__(self, model_dir, model_prototxt, caffe_model, ip, port, capacity, max_episode, push_time_interval, sync_model):
    self.pdtr = latte.Predictor()
    self.pdtr.SetModelFile(model_dir + '/' + model_prototxt, model_dir + '/' + caffe_model)
    self.pdtr.ReadDimConfig(model_dir + '/dim_config.txt')
    self.pdtr.temperature = 1.0
    self.pdtr.sd = [0]*6
    if not sync_model and push_time_interval < 0:
      return
    # Memoire
    self.client = ReplayMemoryClient("tcp://%s:%s" % (ip,str(port+1)), "tcp://%s:%s" % (ip,str(port+2)), capacity)
    rem = self.client.prm
    rem.max_episode = max_episode
    rem.print_info()
    print(rem.rwd_coeff)
    print(rem.cache_flags)
    self.rem = rem
    self.threads = []
    if sync_model:
      # SyncManager
      print("start to sync model from %s:%s" % (ip, str(port)))
      self.threads.append(Thread(target=self.pdtr.sub_worker_main, args=("tcp://%s:%s" % (ip, str(port)), 0))) # 0 for Conn
    # PushWorker
    self.threads.append(Thread(target=self.push_worker_main, args=(push_time_interval,)))
    # start
    for th in self.threads:
      th.daemon = True
      th.start()

  def push_worker_main(self, time_interval):
    cache_count = 0
    if time_interval < 0:
      return
    while True:
      self.client.push_cache()
      cache_count += 1
      #print("pushed caches: %d" % cache_count)
      time.sleep(time_interval)

  def reset(self):
    pass

  def act(self, obs):
    k,x,logp,v = self.pdtr.Predict(obs)
    assert not np.isnan(logp)
    action = [k] + [0]*6
    action[1+2*k+0] = x[0]
    action[1+2*k+1] = x[1]
    return (action,logp,v)

