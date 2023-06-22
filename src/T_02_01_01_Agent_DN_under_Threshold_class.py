# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 23:37:52 2023

@author: thoug
"""

import os
import sys
import gym
import grid2op
from collections import namedtuple

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from grid2op.gym_compat import GymEnv
# from A01_01_Main_Prep import env_out
from gym import Env
from gym.utils.env_checker import check_env
from grid2op.gym_compat import DiscreteActSpace, GymActionSpace
from grid2op.gym_compat import BoxGymObsSpace
from grid2op.Converter import IdToAct
from grid2op.Converter import Converter
from grid2op.Agent import BaseAgent


class train_CEM(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(train_CEM, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),  #4x128
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)  #128*1
        )

    def forward(self, x):
        return self.net(x)
    
    
    def iterate_batches(self,env, net, batch_size):
        batch = []
        action_to_outside = []
        gym_env.reset()
        episode_reward = 0.0
        episode_steps = []
        obs = env.reset()
        # obss = obs[0]
        sm = nn.Softmax(dim=1)
        counter=int()
        terminated = False
        print('start!')

        while True:
            # env.reset()
            # obss = obs
            if type(obs) == np.ndarray:
                obs_v = torch.FloatTensor([obs])
            else:
                obs_v = torch.FloatTensor([gym_env.observation_space.to_gym(obs)])
            # obs_v = torch.FloatTensor([gym_env.observation_space.to_gym(obs)])
            # obs_V = torch.FloatTensor([gym_env.observation_space.to_])
            act_probs_v = sm(net(obs_v))
            act_probs = act_probs_v.data.numpy()[0]
            
            high_act_probs_v_loc = act_probs_v.argmax()
            # act_arr = np.array([act_probs,1-act_probs])
            action = high_act_probs_v_loc.item() # Converte the action number directly
            
            # action_to_outside = action
            # yield action_to_outside
            # action = gym_env.action_space
            next_obs, reward, terminated, *_ = gym_env.step(action) #we need to make the step is also proceed by the action, which is form of sample
            episode_reward += reward
            step = EpisodeStep(observation=obs, action=action)
            episode_steps.append(step)
            obs = next_obs
            
            """
            act_probs_v.argmax() : gives the location of highest probable tensor
            
            gym_env.action_space.from_gym(act_probs_v.argmax()) : do the action based on highest probable tensor  --> replace action
            
            and feed this action into gym_env.step(action)
            
            """
            
            
            if terminated == True:
                # print('terminated')
                gym_env.reset()
                print(counter)
                counter +=1
                e = Episode(reward=episode_reward, steps=episode_steps)
                batch.append(e)
                episode_reward = 0.0
                episode_steps = []
                next_obs = gym_env.reset()
                obs = next_obs
                if len(batch) == batch_size:
                    print('batch_end!')
                    yield batch
                    batch = []
                    gym_env.reset()
                    obs = next_obs
                    
                    
                    
    def filter_batch(self,batch, percentile):
        rewards = list(map(lambda s: s.reward, batch))
        reward_bound = np.percentile(rewards, percentile)
        reward_mean = float(np.mean(rewards))

        train_obs = []
        train_act = []
        for reward, steps in batch:
            if reward < reward_bound:
                continue
            train_obs.extend(map(lambda step: step.observation, steps))
            train_act.extend(map(lambda step: step.action, steps))

        train_obs_v = torch.FloatTensor(train_obs)
        train_act_v = torch.LongTensor(train_act)
        return train_obs_v, train_act_v, reward_bound, reward_mean
    


if __name__ == "__main__":
    
    
    
