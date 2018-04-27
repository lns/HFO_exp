#!/usr/bin/env python
# encoding: utf-8

import latte
import numpy as np
from random import random, randint
import math

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
    return (action, 0.0) # TODO

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
    return (action, 0.0) # TODO

class RoboAgent(object):
  def __init__(self, model_dir, model_prototxt, caffe_model):
    self.pdtr = latte.Predictor()
    self.pdtr.SetModelFile(model_dir + '/' + model_prototxt, model_dir + '/' + caffe_model)
    self.pdtr.ReadLossConfig(model_dir + '/loss_config.txt')
    self.pdtr.temperature = 1.0
    self.pdtr.sd = [0]*6

  def reset(self):
    pass

  def act(self, obs):
    k,x,logp = self.pdtr.Predict(obs)
    assert not np.isnan(logp)
    action = [k] + [0]*6
    action[1+2*k+0] = x[0]
    action[1+2*k+1] = x[1]
    return (action,logp)

