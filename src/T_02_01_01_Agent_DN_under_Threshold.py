# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 18:04:19 2023

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

LEARING_ITERATION = 50
EVAL_EP =1
MAX_EVAL_STEP =10


# env_name = "l2rpn_2019"
# env = grid2op.make(env_name)

# nm_env_train, nm_env_val, nm_env_test = env.train_val_split_random(pct_val=1.0, add_for_test="test", pct_test=1.)
# # 
# print(f"The name of the training environment is {nm_env_train}")
# print(f"The name of the validation environment is {nm_env_val}")
# print(f"The name of the test environment is {nm_env_test}")

"""
 “set_line_status”, “change_line_status”, “set_bus” and “change_bus”.
"""



from grid2op.gym_compat import GymEnv
from A01_01_Main_Prep import env_out
from gym import Env
from gym.utils.env_checker import check_env


training_env_name = "l2rpn_2019_train"
training_env = grid2op.make(training_env_name)

test_env_name = "l2rpn_2019_test"

gym_env = GymEnv(env_out)


# isinstance(gym_env, Env)  --> to check if the environment is actually GYM env

act_space_attr_to_keep = ["change_bus"]
obs_space_attr_to_keep = ["gen_p", "gen_q", "gen_v", "load_p", "load_q", "load_v", "p_or", "q_or", "v_or", "a_or", "p_ex", "q_ex", "v_ex", "a_ex", "rho", "topo_vect", "line_status", "timestep_overflow"]

from grid2op.gym_compat import DiscreteActSpace, GymActionSpace
from grid2op.gym_compat import BoxGymObsSpace
from grid2op.Converter import IdToAct
from grid2op.Converter import Converter

# class Converter_here(Converter):
#     def __init__(self, action_space):
#         ActionSpace.__init__(
#             self, action_space
#         )
        
        

gym_env.action_space = DiscreteActSpace(env_out.action_space, attr_to_keep=act_space_attr_to_keep) 
# gym_env.action_space = DiscreteActSpace(env_out.action_space)  #giving how much action space is available, for now 427
bb = gym_env.action_space


# converted_action_space = Converter_here.convert_action_to_gym(training_env.action_space())
# converted_action_space = Converter.convert_action_to_gym(training_env.action_space())
# gym_env.action_space = GymActionSpace(training_env.action_space)




gym_env.observation_space = BoxGymObsSpace(env_out.observation_space, attr_to_keep=obs_space_attr_to_keep)

# aa = gym_env.action_space

obs = gym_env.reset()



# my_agent = AgentFromGym(training_env,)


#%%

N_ACTION = 157
HIDDEN_SIZE = 300
OBS_SIZE = 324
PERCENTILE = 75
BATCH_SIZE = 20
counter = 0


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
    
Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])

class DiscreteOneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(DiscreteOneHotWrapper, self).__init__(env)
        assert isinstance(env.observation_space,
                          gym.spaces.Discrete)
        shape = (env.observation_space.n, )
        self.observation_space = gym.spaces.Box(
            0.0, 1.0, shape, dtype=np.float32)

    def observation(self, observation):
        res = np.copy(self.observation_space.low)
        res[observation] = 1.0
        return res


def iterate_batches(env, net, batch_size):
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
        

        


def filter_batch(batch, percentile):
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
    
    
    
    
"""
https://github.com/rte-france/Grid2Op/blob/master/getting_started/04_TrainingAnAgent.ipynb
"""


#%%


if __name__ == "__main__":
    env = env_out
    env.reset()
    # envenv = grid2op.make(env_name)
    # env = gym.wrappers.Monitor(env, directory="mon", force=True)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = train_CEM(OBS_SIZE, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=1e-4)
    
    path_name = "C:\\Users\\thoug\\OneDrive\\SS2023\\Internship\\02_G2OP\\File\\Agent\\June"
    
    """
    one : prototype, 
    two : trained with environment parametered
    three : trained with environment parameter and L2RPN Reward with solved at 30000
    four : trained with environment parameter and L2RPN Reward with solved at 90000
    
    six : same as four with solved at 120000
    seven : updated thermal limit based on the paper, solved based on loss value not reward. solved when loss<0.00001
   
    eight : seven with writer function on (let's see if it work)
    """
    
    
    path_1 = os.path.join(path_name, 'eight')
    path_2 = os.path.join(path_name, 'eight_entire')
    
    path_to_save = os.path.join(path_name, 'save_8')
    
    writer = SummaryWriter(comment="-Agent_8")

    trainend = True

    if trainend == False:

        for iter_no, batch,  in enumerate(iterate_batches(env, net, BATCH_SIZE)):
            obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
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
            if loss_v.item() < 0.00001:
                print("Solved!")
                
                torch.save(net.state_dict(), path_1)
                torch.save(net, path_2)
                
                break
        writer.close()
    #%%
    else:
        
        
        from grid2op.Action import TopologyAction, SerializableActionSpace
        
        net = torch.load(path_2)
        # net.eval()
        
        my_agent=AgentFromGym(gym_env, net)
        
        # env = grid2op.make(test_env_name,  
        #                    action_class=TopologyAction) #for testing environment with limited Chronics
        
        # env = grid2op.make(env_name,
        #                    action_class = TopologyAction) #actual environment with every Chronics 
        
        # env = grid2op.make("rte_case14_realistic",
        #                    action_class = TopologyAction) #to compare with WS2022 seminar result
        
        env = env_out
        
        
        from grid2op.Runner import Runner
        from grid2op.Reward import L2RPNReward
        from tqdm import tqdm
        
        runner = Runner(**env.get_params_for_runner(),
                agentClass=None,
                agentInstance=my_agent
                )
        res = runner.run(path_save=path_to_save,
                          nb_episode=1000, 
                          # episode_id =[274],
                          nb_process=10,
                          pbar=True)
        print("The results for the my agent are:")
        for _, chron_id, cum_reward, nb_time_step, max_ts in res:
            msg_tmp = "\tFor chronics with id {}\n".format(chron_id)
            msg_tmp += "\t\t - cumulative reward: {:.6f}\n".format(cum_reward)
            msg_tmp += "\t\t - number of time steps completed: {:.0f} / {:.0f}".format(nb_time_step, max_ts)
            print(msg_tmp)
            
            
            
        # from grid2op.Agent import DoNothingAgent
        # #DN
        # runner_DN = Runner(**env.get_params_for_runner(),
        #                 agentClass=DoNothingAgent
        #                 )
        # res_DN = runner_DN.run(nb_episode=1 ,episode_id =[274],nb_process=2, pbar = tqdm )

        # print("The results for DoNothing agent are:")
        # for _, chron_name, cum_reward, nb_time_step, max_ts in res_DN:
        #     msg_tmp_DN = "\tFor chronics with id {}\n".format(chron_name)
        #     msg_tmp_DN += "\t\t - cumulative reward: {:.6f}\n".format(cum_reward)
        #     msg_tmp_DN += "\t\t - number of time steps completed: {:.0f} / {:.0f}".format(nb_time_step, max_ts)
        #     print(msg_tmp_DN)
            
            
        # #Power Line Switch Agent
        # from grid2op.Agent import PowerLineSwitch
        # runner_PLS = Runner(**env.get_params_for_runner(),
        #                 agentClass=PowerLineSwitch,
        #                 )
        # res_PLS = runner_PLS.run(nb_episode=1 ,episode_id =[274],nb_process=2, pbar = tqdm )
        # print("The results for the PowerLineSwitch agent are:")
        # for _, chron_name, cum_reward, nb_time_step, max_ts in res_PLS:
        #     msg_tmp_PLS = "\tFor chronics with id {}\n".format(chron_name)
        #     msg_tmp_PLS += "\t\t - cumulative reward: {:.6f}\n".format(cum_reward)
        #     msg_tmp_PLS += "\t\t - number of time steps completed: {:.0f} / {:.0f}".format(nb_time_step, max_ts)
        #     print(msg_tmp_PLS)

            
            
            
 #%%           
        import os
        import grid2op
        from grid2op.Episode import EpisodeData
        import plotly.graph_objects as go
        import plotly.io as pio
        pio.renderers.default = 'png'
        os.environ['KMP_DUPLICATE_LIB_OK']='True'
        
        path_name = "C:\\Users\\thoug\\OneDrive\\SS2023\\Internship\\02_G2OP\\File\\Agent\\June"
        path_to_save = os.path.join(path_name, 'save_8')

        episode_studied_res = EpisodeData.list_episode(path_to_save)
#%%

        for a in range(len(episode_studied_res)):
            this_episode_res = EpisodeData.from_disk(*episode_studied_res[a])
            episode_data_res = this_episode_res
            
            # for act in this_episode_DN.actions:print(act)
            #     
            
            val_lgen3 = np.zeros(len(this_episode_res.observations))
        
        
        





            line_disc = 0
            line_reco = 0
            line_changed = 0
            obj_mod=0
            sub_mod = 0
            test = dict()
            sub_id =dict()
            
            for act in this_episode_res.actions:
                dict_ = act.as_dict()
                if "change_bus_vect" in dict_:
                    obj_mod += dict_['change_bus_vect']['nb_modif_objects']
                    sub_mod += dict_['change_bus_vect']['nb_modif_subs']
                    # test = dict_['change_bus_vect']['1'].append(dict_['change_bus_vect']['1'])
                    # sub_id = dict_['change_bus_vect']['modif_subs_id'].append(dict_['change_bus_vect']['modif_subs_id'])
            print(f'The topology changed of MY Agent')
            # print(f'Total lines set to connected : {line_reco}')
            # print(f'Total lines set to disconnected : {line_disc}')
            # print(f'Total lines changed: {line_changed}')
            print(f'Total number of changed objects: {obj_mod}')
            print(f'Total number of changed substation: {sub_mod}')
            
            
            """
            
            dict_['change_bus_vect']
            Out  [6]: {'nb_modif_objects': 2, '1': {'2': {'type': 'line (origin)'}, '4': {'type': 'line (origin)'}}, 'nb_modif_subs': 1, 'modif_subs_id': ['1']}
            
            dict_['change_bus_vect'].keys()
            Out  [7]: dict_keys(['nb_modif_objects', '1', 'nb_modif_subs', 'modif_subs_id'])
            
            """
                            
        from grid2op.Episode import EpisodeReplay
        plot_epi_MY = EpisodeReplay(path_to_save)
        # plot_epi_MY.replay_episode(res[0][1], gif_name="episode_MY")
                        

        print("-----------------------DN AGENT OVER------------------------")
#%%        
        # env.seed(0)  # for reproducible experiments
        # episode_count = 2  # i want to make 100 episodes
        
        # ###################################
        # THE_CHRONIC_ID = 179
        # ###################################
        
        # # i initialize some useful variables
        # reward = 0
        # done = False
        # total_reward = 0
        
        # # and now the loop starts
        # for i in range(episode_count):
        #     ###################################
        #     env.set_id(THE_CHRONIC_ID)
        #     ###################################
        
        #     ob = env.reset()
        
        #     # now play the episode as usual
        #     while True:
        #        action = my_agent.act(ob, reward, done)
        #        ob, reward, done, info = env.step(action)
        #        total_reward += reward
        #        if done:
        #            # in this case the episode is over
        #            break
        
        # # Close the env and write monitor result info to disk
        # env.close()
        # print("The total reward was {:.2f}".format(total_reward))
        


"""
https://pytorch.org/tutorials/beginner/saving_loading_models.html
"""

#%%
#try read the issues

# tensorboard dev upload --logdir "C:\Users\thoug\OneDrive\SS2023\Internship\02_G2OP\runs\Jun08_03-08-14_BOOK-5K4M42628E-Agent_8"