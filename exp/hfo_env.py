#!/usr/bin/env python
# encoding: utf-8

# Before running this program, first Start HFO server:
# $> ./bin/HFO --offense-agents 1 --headless --port port --untouched-time 100 --frames-per-trial 500

import os
#import threading
#import itertools
from threading import Thread
from hfo import *
import math, time
from random import random, randint
import subprocess
from hfo_config import *
from hfo_agent import *
#
from memoire import ReplayMemory, ReplayMemoryServer, ReplayMemoryClient, Bind, Conn
import numpy as np

class HFOEnv():
    def __init__(self, port):
        untouched_time = 100  # end the game if no one touch the ball in untouched-time frames
        self.frames_per_trial = 500  # maximal length of game
      
        cmd = hfo_path + "/bin/HFO --offense-agents 1 --fullstate --headless --port {} --untouched-time {} --frames-per-trial {}".format(port, untouched_time, self.frames_per_trial)
        print('Starting server with command: %s' % cmd)
        self.server_process = subprocess.Popen(cmd.split(' '), shell=False)
        self.hfo = HFOEnvironment()
        self.init = False
        self.port = port
        self.observation_shape = [59]
        time.sleep(3)
        self.action_shape = [3,2] # Dash, Turn, Kick: 2 + 2 + 2

    def reset(self):
        if not self.init:
            self.hfo.connectToServer(LOW_LEVEL_FEATURE_SET,
                      hfo_path+'/bin/teams/base/config/formations-dt', self.port,
                      'localhost', 'base_left', False)
            self.init = True
        self.status = IN_GAME
        while True:
            obs = self.hfo.getState()
            obs[-1] = 0
            if self.status==IN_GAME:
                if obs[50]==1:
                    break
            self.status = self.hfo.step()
        self.obs=obs
        self.got_kickable_reward=False
        self.ball_proximity, self.ball_dist_goal, self.kickable = self.ext(obs)
        self.kickable = False
        self.episode_step = 1
        return obs

    def push_action(self, action):
        action_map = {0:DASH,1:TURN,2:KICK,3:TACKLE}  # Tackle cannot use in 1 offense mode
        action_shape = (3,2,2,2) # 3 action type, 2 + 2 + 2
        if action[0] == 0:
            angle = math.atan2(action[1],action[2])/math.pi*180
            power = min(math.sqrt(action[1]**2 + action[2]**2), 1)*100
            self.hfo.act(DASH, power, angle)
        if action[0] == 1:
            angle = math.atan2(action[3],action[4])/math.pi*180
            self.hfo.act(TURN, angle)
        if action[0] == 2:
            angle = math.atan2(action[5],action[6])/math.pi*180
            power = min(math.sqrt(action[5]**2 + action[6]**2), 1)*100
            self.hfo.act(KICK, power, angle)
        if action[0] == 3:
            self.hfo.act(TACKLE, 0)

    def ext(self, obs):
        ball_proximity = obs[53]
        ball_dist = 1 - ball_proximity
        goal_dist = 1 - obs[15]
        kickable = obs[12]==1 and not self.got_kickable_reward
        ball_angle = math.atan2(obs[51], obs[52])
        goal_angle = math.atan2(obs[13], obs[14])
        ball_dist_goal = math.sqrt(ball_dist*ball_dist + goal_dist*goal_dist -
                            2.*ball_dist*goal_dist*math.cos(ball_angle - goal_angle))
        #ball_dist_goal = math.sqrt(ball_dist*ball_dist + goal_dist*goal_dist -
        #                     2.*ball_dist*goal_dist*(obs[52]*obs[14]+obs[51]*obs[13]))
        return ball_proximity, ball_dist_goal, kickable

    def reward(self, obs):
        ball_proximity, ball_dist_goal, kickable = self.ext(obs)
        ret = (ball_proximity - self.ball_proximity) - 3.0*(ball_dist_goal - self.ball_dist_goal) + kickable
        if obs[50]==1:
            if kickable:
                self.got_kickable_reward=True
            self.ball_proximity, self.ball_dist_goal, self.kickable = ball_proximity, ball_dist_goal, kickable
        else:
            ret = 0
        return ret

    def step(self,action):
        if self.status == IN_GAME:
            self.push_action(action)
            self.status = self.hfo.step()
            if self.status == IN_GAME:
                obs = self.hfo.getState()
                succ = obs[-1]
                #obs = obs[:-1]
                obs[-1] = float(self.episode_step)/self.frames_per_trial
                rwd = self.reward(obs)
            else:
                obs = []
                succ = 0
                if self.status == GOAL:
                    rwd = 5
                else:
                    rwd = 0
            terminal = self.status!=IN_GAME
            info = [self.status, succ]
            self.episode_step += 1
            if self.status == SERVER_DOWN:
                self.hfo.act(QUIT)
            return obs, rwd, terminal, info
        else:
            print('Game Over, status={} . PLS reset the game!'.format(self.status))
            return -1

def transform_state(rem, obs):
    buf = np.getbuffer(obs)
    if rem.state_size != obs.size * 4: # sizeof(uint8), sizeof(float)
      print "%d != %d" % (rem.state_size, obs.size * 4)
    assert(rem.state_size == obs.size * 4)
    assert(rem.state_size == len(buf))
    assert(obs.dtype == np.float32)
    # Reinterpret as np.uint8 array
    s = np.frombuffer(buf, dtype=np.uint8)
    return s

def transform_label(rem, act, rwd, logp):
    assert(len(act) == 7)
    a = np.ndarray((rem.action_size),dtype=np.float32)
    a[0] = act[0]
    assert(a[0] < 3)
    a[1] = act[1+int(round(a[0]))*2 + 0]
    a[2] = act[1+int(round(a[0]))*2 + 1]
    #a_norm = math.sqrt(a[1]**2 + a[2]**2)
    #if a_norm > 1:
    #  a[1] /= a_norm
    #  a[2] /= a_norm
    r = np.ndarray((rem.reward_size),dtype=np.float32)
    r[0] = rwd
    p = np.ndarray((rem.prob_size),dtype=np.float32)
    p[0] = logp
    return (a,r,p)

def train(mdl_file, port, rem_port, capacity, max_episode, push_time_interval, sync_model, mode, epsilon):
    print([IN_GAME, GOAL, CAPTURED_BY_DEFENSE, OUT_OF_BOUNDS, OUT_OF_TIME, SERVER_DOWN])
    # Agent
    ip = "100.102.32.6"
    agt = RoboAgent(".", "net_hfo_test.prototxt", mdl_file, ip=ip, port=rem_port,
        capacity=capacity, max_episode=max_episode, push_time_interval=push_time_interval, sync_model=sync_model)
    #offagt = RoboAgent(".", "net_hfo_test.prototxt", "il%3.1f.caffemodel" % epsilon, ip='NULL', port=0,
    #    capacity=0, max_episode=0, push_time_interval=-1, sync_model=0)
    rule_agt = RuleAgent()
    if mode == 'IL':
      agt.pdtr.sd = [0.2]*6
      agt.pdtr.temperature = 1.0
    elif mode == 'RL':
      agt.pdtr.sd = [0.2]*6
      agt.pdtr.temperature = 1.0
    else:
      print('Unknown mode: %s' % mode)
    # Memoire
    client = agt.client
    rem = client.prm
    s = np.ndarray((rem.state_size), dtype=np.uint8)
    a = np.ndarray((rem.action_size),dtype=np.float32)
    r = np.ndarray((rem.reward_size),dtype=np.float32)
    #
    hfo=HFOEnv(int(port))
    obs = hfo.reset()
    agt.reset()
    rule_agt.reset()
    game_num=0
    rem.new_episode()
    while True:
      s = transform_state(rem, obs)
      if random() > epsilon:
        action,logp,v = agt.act(obs)
        logp += math.log(1-epsilon)
      else:
        action,logp,v = rule_agt.act(obs)
        logp += math.log(epsilon)
      #_, est_logp, _ = offagt.act(obs)
      obs,rwd,terminal,info = hfo.step(action)
      a,r,p = transform_label(rem, action, rwd, logp)
      rem.add_entry(s, a, r, logp, v, 1.0) # logp or est_logp
      if terminal:
        rem.close_episode()
        client.update_counter()
        obs = hfo.reset()
        agt.reset()
        rule_agt.reset()
        game_num+=1
        rem.new_episode()
    os.system("killall -9 rcssserver")

def test(mdl_file, port, rem_port, capacity, max_episode, push_time_interval, sync_model, mode, epsilon):
  # Agent
    ip = "100.102.32.6"
    agt=RoboAgent(".", "net_hfo_test.prototxt", mdl_file, ip=ip, port=rem_port,
        capacity=capacity, max_episode=max_episode, push_time_interval=push_time_interval, sync_model=sync_model)
    #agt.pdtr.scale = [1.0,0.1,1.0]
    agt.pdtr.sd = [0.0]*6
    agt.pdtr.temperature = 1.0
    #
    hfo=HFOEnv(int(port))
    obs = hfo.reset()
    agt.reset()
    game_num=0
    while True:
        action,logp,v = agt.act(obs)
        obs,rwd,terminal,info = hfo.step(action)
        if terminal:
            obs = hfo.reset()
            agt.reset()
            game_num+=1
    os.system("killall -9 rcssserver")

if __name__ == '__main__':
  import sys
  if len(sys.argv) < 7:
    print("Usage: %s [train|test] model_file port rem_port sync_model mode epsilon" % sys.argv[0])
    exit(0)
  if sys.argv[1] == 'train':
    train(mdl_file=sys.argv[2], port=sys.argv[3], rem_port=int(sys.argv[4]),
        capacity=65536, max_episode=32, push_time_interval=1.0, sync_model=int(sys.argv[5]), mode=sys.argv[6],
        epsilon=float(sys.argv[7]))
  elif sys.argv[1] == 'test':
    test(mdl_file=sys.argv[2], port=sys.argv[3], rem_port=int(sys.argv[4]),
        capacity=65536, max_episode=32, push_time_interval=-1.0, sync_model=int(sys.argv[5]), mode=sys.argv[6],
        epsilon=float(sys.argv[7]))

