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
from A01_01_Main_Prep import env_out_l2rpn as env_out
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
    
    
class training():
    def __init__(self, ENV, NET, BATCH_SIZE, BATCH=None, PERCENTILE=75):
        self.env = ENV
        self.net = NET
        self.batch_size = BATCH_SIZE
        self.batch = BATCH
        self.percentile = PERCENTILE
    
    def iterate_batches(self):
        batch = []
        action_to_outside = []
        gym_env.reset()
        episode_reward = 0.0
        episode_steps = []
        obs = self.env.reset()
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
            act_probs_v = sm(self.net(obs_v))
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
                if len(batch) == self.batch_size:
                    print('batch_end!')
                    self.batch = batch
                    yield batch
                    batch = []
                    gym_env.reset()
                    obs = next_obs
                    
                    
                    
    def filter_batch(self):
        rewards = list(map(lambda s: s.reward, self.batch))
        reward_bound = np.percentile(rewards, self.percentile)
        reward_mean = float(np.mean(rewards))

        train_obs = []
        train_act = []
        for reward, steps in self.batch:
            if reward < reward_bound:
                continue
            train_obs.extend(map(lambda step: step.observation, steps))
            train_act.extend(map(lambda step: step.action, steps))

        train_obs_v = torch.FloatTensor(train_obs)
        train_act_v = torch.LongTensor(train_act)
        return train_obs_v, train_act_v, reward_bound, reward_mean
    


if __name__ == "__main__":
    
    N_ACTION = 157
    HIDDEN_SIZE = 300
    OBS_SIZE = 324
    PERCENTILE = 75
    BATCH_SIZE = 20
    counter = 0
    
    Episode = namedtuple('Episode', field_names=['reward', 'steps'])
    EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])
    
    
    env = env_out
    env.reset()
    # envenv = grid2op.make(env_name)
    # env = gym.wrappers.Monitor(env, directory="mon", force=True)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    
    act_space_attr_to_keep = ["change_bus"]
    obs_space_attr_to_keep = ["gen_p", "gen_q", "gen_v", "load_p", "load_q", "load_v", "p_or", "q_or", "v_or", "a_or", "p_ex", "q_ex", "v_ex", "a_ex", "rho", "topo_vect", "line_status", "timestep_overflow"]
    
    
    gym_env = GymEnv(env_out)
    gym_env.action_space = DiscreteActSpace(env_out.action_space, attr_to_keep=act_space_attr_to_keep) 
    gym_env.observation_space = BoxGymObsSpace(env.observation_space, attr_to_keep=obs_space_attr_to_keep)
    
    # obs = gym_env.reset()
    
    
    net = train_CEM(OBS_SIZE, HIDDEN_SIZE, n_actions)

    t = training(gym_env, net, BATCH_SIZE)
    
    
    
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=1e-4)
    
    path_name = "C:\\Users\\thoug\\OneDrive\\SS2023\\Internship\\04_Code\\CEM_Agent_G2OP\\File\\Agent\\July"
    
    """
    one : prototype,

    """
    
    
    path_1 = os.path.join(path_name, 'one')
    path_2 = os.path.join(path_name, 'one_entire')
    
    path_to_save = os.path.join(path_name, 'save_1')
    
    writer = SummaryWriter(comment="-Agent_1")

    trainend = False

    if trainend == False:

        for iter_no, batch,  in enumerate(t.iterate_batches()):
            obs_v, acts_v, reward_b, reward_m = t.filter_batch()
            optimizer.zero_grad()
            action_scores_v = net(obs_v)
            loss_v = objective(action_scores_v, acts_v)  #objective(current, target)
            loss_v.backward()
            optimizer.step()
            print("%d: loss=%.6f, reward_mean=%.1f, rw_bound=%.1f" % (
                iter_no, loss_v.item(), reward_m, reward_b))
            writer.add_scalar("loss", loss_v.item(), iter_no)
            writer.add_scalar("reward_bound", reward_b, iter_no)
            writer.add_scalar("reward_mean", reward_m, iter_no)
            if loss_v.item() < 0.0001:
                print("Solved!")
                
                torch.save(net.state_dict(), path_1)
                torch.save(net, path_2)
                
                break
        writer.close()
    
    
