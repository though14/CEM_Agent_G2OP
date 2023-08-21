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
from A01_01_Main_Prep import env_out_rte as env_out
from A01_01_Main_Prep import env_out_rte_test as env_test
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
            
            
            rd_rate = np.random.random()
            
            if rd_rate <= 5.0:
            
                # high_act_probs_v_loc = act_probs_v.argmax()
                # act_arr = np.array([act_probs,1-act_probs])
                # action = high_act_probs_v_loc.item() # Converte the action number directly
                action = np.random.choice(len(act_probs), p=act_probs)
                
            else:
                action = np.random.randint(n_actions)
            
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
                # gym_env.reset()
                print(f'{counter}-- {episode_reward}')
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

from grid2op.Agent import BaseAgent

class AgentFromGym(BaseAgent):
    def __init__(self, gym_env, neural_network):
        self.gym_env = gym_env
        self.neural_network = neural_network
        self.sm = nn.Softmax(dim=1)
        self.do_nothing = self.gym_env.init_env.action_space({})
        BaseAgent.__init__(self, gym_env.init_env.action_space)
        # self.trained_aget = trained_agent
    def act(self, obs, reward, done):
        
        # gym_obs = self.gym_env.observation_space.to_gym(obs)
        
        gym_obs= obs
        
        if type(gym_obs) == np.ndarray:
            gym_obs_v = torch.FloatTensor([gym_obs])
        else:
            gym_obs_v = torch.FloatTensor([gym_env.observation_space.to_gym(gym_obs)])
            
        agent_act_probs_v = self.sm(self.neural_network(gym_obs_v))
        agent_act_probs = agent_act_probs_v.data.numpy()[0]
        
        agent_high_act_probs_v_loc = agent_act_probs_v.argmax()
        agent_action = agent_high_act_probs_v_loc.item()
        
        
        if np.any(obs.rho>0.95):
            grid2op_act = self.gym_env.action_space.from_gym(agent_action)
        else:
            grid2op_act = self.do_nothing
        
        
        
        
        # gym_act = self.action_from_CEM
        # grid2op_act = self.gym_env.action_space.from_gym(gym_act)
        return grid2op_act    


if __name__ == "__main__":
    
    N_ACTION = 157
    HIDDEN_SIZE = 300
    OBS_SIZE = 324
    PERCENTILE = 95
    BATCH_SIZE = 100
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

    t = training(gym_env, net, BATCH_SIZE, PERCENTILE)
    
    
    
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=1e-4)
    
    path_name = "C:\\Users\\thoug\\OneDrive\\SS2023\\Internship\\04_Code\\CEM_Agent_G2OP\\File\\Agent\\Aug_rte_rand"
    
    """
    one : prototype, Batch_size = 100, 95% filter, random action 2%
    two : Batch_size = 50, 80% filter, random action 3%
    three : Batch_size = 50, 90% filter, no random
    Four : Batch_size =100, 95% filter
    Five : same as four

    """
    
    
    path_1 = os.path.join(path_name, 'fve')
    path_2 = os.path.join(path_name, 'five_entire')
    
    path_to_save = os.path.join(path_name, 'save_5')
    
    writer = SummaryWriter(comment="-Agent_5_rte_rand")

    trainend = True

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
            # if loss_v.item() < 0.0001:
            if reward_m > 30000:
                print("Solved!")
                
                torch.save(net.state_dict(), path_1)
                torch.save(net, path_2)
                
                break
        writer.close()
    
    else:
        
        
        from grid2op.Action import TopologyAction, SerializableActionSpace, TopologyChangeAndDispatchAction
        
        net = torch.load(path_2)
        # net.eval()
        # env = env_test_no_p
        my_agent=AgentFromGym(gym_env, net)
        

        
        # env = env_out
        # env = grid2op.make('l2rpn_2019_test')
        # env_test = grid2op.make('l2rpn_2019_test')
        
        env_test = env_test 
        
        from grid2op.Runner import Runner
        from grid2op.Reward import L2RPNReward
        from tqdm import tqdm
        
        runner = Runner(**env_test.get_params_for_runner(),
                agentClass=None,
                agentInstance=my_agent
                )
        res = runner.run(path_save=path_to_save,
                          nb_episode=10, 
                          # episode_id =[274],
                          nb_process=1,
                          pbar=True)
        print("The results for the my agent are:")
        for _, chron_id, cum_reward, nb_time_step, max_ts in res:
            msg_tmp = "\tFor chronics with id {}\n".format(chron_id)
            msg_tmp += "\t\t - cumulative reward: {:.6f}\n".format(cum_reward)
            msg_tmp += "\t\t - number of time steps completed: {:.0f} / {:.0f}".format(nb_time_step, max_ts)
            print(msg_tmp)
            
            
        #%%  incase for DN Agent comparasion

        # from grid2op.Agent import DoNothingAgent
        # #DN
        # runner_DN = Runner(**env_test.get_params_for_runner(),
        #                 agentClass=DoNothingAgent
        #                 )
        # res_DN = runner_DN.run(nb_episode=10 ,nb_process=1, pbar = True)

        # print("The results for DoNothing agent are:")
        # for _, chron_name, cum_reward, nb_time_step, max_ts in res_DN:
        #     msg_tmp_DN = "\tFor chronics with id {}\n".format(chron_name)
        #     msg_tmp_DN += "\t\t - cumulative reward: {:.6f}\n".format(cum_reward)
        #     msg_tmp_DN += "\t\t - number of time steps completed: {:.0f} / {:.0f}".format(nb_time_step, max_ts)
        #     print(msg_tmp_DN)
            
