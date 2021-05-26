"""import argparse
import pickle 
import numpy as np

parser = argparse.ArgumentParser(description='...')
#parser.add_argument('-l','--layer', type=int, nargs='+', required=True, action='append', help='layer list')
parser.add_argument("--activation", nargs="+", default= ["relu"])

args = parser.parse_args()

#print(args.layer)
print(args.activation)
cpt = 1

string1 =  'data4/listfile_40_'+str(cpt)+'.data' #_evol'+ , _pos'+
with open(string1, 'rb') as filehandle:
# read the data as binary data stream
    lst = pickle.load(filehandle)


string2 = 'data4/nei_tab_pos_40_'+str(cpt)+'.data'
with open(string2, 'rb') as filehandle:
    # read the data as binary data stream
    nei_tab = pickle.load(filehandle)

print(np.shape(nei_tab)) # good (40,20)
print(np.shape(lst)) # good (40,20)


print(lst[19][19])"""

#!/usr/bin/env python
# encoding: utf-8

import gym
import gym_example
import pickle
import argparse

import ray
from ray import tune
from ray.rllib.agents import ppo
from ray.rllib.agents import ddpg
from ray.rllib.agents import a3c
import ray.rllib.agents.impala as impala
import gym
import gym_example
from ray.tune.registry import register_env


from ray.tune import grid_search
#from my_env import ContentCaching
import pickle
import time
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt 
from gym_example.envs.caching_env import Caching_v0


def ret_lst(cpt):
    string1 =  'data4/listfile_40_'+str(cpt)+'.data' #_evol'+ , _pos'+
    with open(string1, 'rb') as filehandle:
    # read the data as binary data stream
        lst = pickle.load(filehandle)
    return lst

def ret_nei(cpt):
    string2 = 'data4/nei_tab_pos_40_'+str(cpt)+'.data'
    with open(string2, 'rb') as filehandle:
        # read the data as binary data stream
        nei_tab = pickle.load(filehandle)
    return nei_tab

def train( algo, config_train, stop_criteria):
    """
    Train an RLlib IMPALA agent using tune until any of the configured stopping criteria is met.
        See https://docs.ray.io/en/latest/tune/api_docs/execution.html#tune-run
    :return: Return the path to the saved agent (checkpoint) and tune's ExperimentAnalysis object
        See https://docs.ray.io/en/latest/tune/api_docs/analysis.html#experimentanalysis-tune-experimentanalysis
    """
    if algo == "ppo":
        analysis = ray.tune.run(ppo.PPOTrainer, config=config_train, local_dir="~/ray_results", stop=stop_criteria,
                           checkpoint_at_end=True)
  

    lr = analysis.get_best_config(metric='episode_reward_mean', mode="max")["lr"] 
    fc_hid = analysis.get_best_config(metric='episode_reward_mean', mode="max")["model"]["fcnet_hiddens"] 
    fc_act = analysis.get_best_config(metric='episode_reward_mean', mode="max")["model"]["fcnet_activation"] 

    # list of lists: one list per checkpoint; each checkpoint list contains 1st the path, 2nd the metric value
    checkpoints = analysis.get_trial_checkpoints_paths(trial=analysis.get_best_trial('episode_reward_mean', mode = 'max'),
                                                       metric='episode_reward_mean')
    # retriev the checkpoint path; we only have a single checkpoint, so take the first one

    checkpoint_path = checkpoints[0][0]
    print("Checkpoint path:", checkpoint_path)
    return checkpoint_path, analysis, lr, fc_hid, fc_act


def test(config_test, algo, path, lr, fc_hid, fc_act):

    """Test trained agent for a single episode. Return the episode reward"""
    # instantiate env class
    unused_shared = []
    unused_own = []
    unsatisfied_shared = []
    unsatisfied_own = []

    episode_reward = 0
    """
    self.config_test["num_workers"] = 0
    self.config_test["lr"] = lr
    self.config_test['model']["fcnet_hiddens"] = fc_hid
    self.config_test['model']["fcnet_activation"] = fc_act
    """

    if algo == "ppo":
        agent = ppo.PPOTrainer(config=config_test)
    
    agent.restore(path)

    #env = self.agent.workers.local_worker().env
    #env = self.env_class(self.env_config)
    #env = ContentCaching(*self.config_train)
    #env = self.config_train["env"]#env_config)
    #env = self.env_class(3)
    #env = ContentCaching
    #env = self.env
    #self.env = ContentCaching
    #env = self.config_train["env"]
    env = gym.make("caching-v0", config=config_test["env_config"])
    
 
    obs = env.reset()
    done = False

    while not done:
        action = agent.compute_action(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward

        unused_shared.append(info["unused_shared"])
        unused_own.append(info["unused_own"])
        unsatisfied_shared.append(info["unsatisfied_shared"])
        unsatisfied_own.append(info["unsatisfied_own"])
    
    return episode_reward, unused_shared, unused_own, unsatisfied_shared, unsatisfied_own




if __name__ == "__main__":

    config = {
        "env": "caching-v0",  # or "corridor" if registered above
        "env_config": {
            "ttl_var": 3,
            "variable": [8,8,8,4],
            "nei_tab": ret_nei(1),
            "lst_tab": ret_lst(1),
                   
        },
      

        "model": {
            # By default, the MODEL_DEFAULTS dict above will be used.

            # Change individual keys in that dict by overriding them, e.g.
            "fcnet_hiddens": [128, 128, 128],
            "fcnet_activation": "relu",
            "vf_share_layers": True,
        },

        "lr": 1e-2,  # try different lrs
        "num_workers": 2,  # parallelism
        "framework": "torch"# if args.torch else "tf",
    }
    stop_criteria = {
                    "training_iteration": 1,#args.stop_iters,
                    "timesteps_total": 990000000,#args.c,
                    "episode_reward_mean": 0.0000000001#args.stop_reward,
                    }

    #print("mmmmmmmmmmmmmmmmmmmmmm = ", config["env_config"]["env"])
    #"""
    select_env = "caching-v0"
    register_env(select_env, lambda config: Caching_v0(config))
    
    checkpoint_path, results, lr, fc_hid, fc_act = train("ppo", config, stop_criteria)
    print("gym.make successfully")


    reward, unused_shared ,unused_own, unsatisfied_shared, unsatisfied_own  = test(config, "ppo" ,checkpoint_path, lr, fc_hid, fc_act)
	#"""


    #env = gym.make("caching-v0", config=config["env_config"])
    #env.reset()
    #print("gym.make successfully")


