import argparse

import ray
from ray import tune
from ray.rllib.agents import ppo
from my_env import ContentCaching
import pickle
import time
import numpy as np


parser = argparse.ArgumentParser()

parser.add_argument("--stop-iters", type=int, default= 1)#50)
parser.add_argument("--stop-timesteps", type=int, default=100000)
parser.add_argument("--stop-reward", type=float, default=0.1)

def ret_lst(cpt):
    string1 =  'data/listfile_evol'+str(cpt)+'.data' #_evol'+ , _pos'+
    with open(string1, 'rb') as filehandle:
    # read the data as binary data stream
        lst = pickle.load(filehandle)
    return lst

def ret_nei(cpt):
    string2 = 'data/nei_tab_pos'+str(cpt)+'.data'
    with open(string2, 'rb') as filehandle:
        # read the data as binary data stream
        nei_tab = pickle.load(filehandle)
    return nei_tab

class customExperimentClass():

    def __init__(self,ttl_var, cpt, variable, stop_iters=1, stop_timesteps=100000, stop_reward=0.1):#
        #ray.shutdown()
        #ray.init(num_cpus=3)#num_cpus=2, num_gpus=0)
        self.env = ContentCaching#gym.make("ContentCaching-v0")
        self.config_train = {
                        "env": ContentCaching,
                        "env_config": {
                        "ttl_var": ttl_var,
                        "variable": variable,#[8,8,8,4],
                        "nei_tab": ret_nei(cpt),
                        "lst_tab": ret_lst(cpt),

        }}
        self.config_test = {
                        "env": ContentCaching,
                        "env_config": {
                        "ttl_var": ttl_var,
                        "variable": variable,
                        "nei_tab": ret_nei(5),
                        "lst_tab": ret_lst(5),

        }}
        self.save_dir = "~/ray_results"
        self.stop_criteria = {
                    "training_iteration": stop_iters,#args.stop_iters,
                    "timesteps_total": stop_timesteps,#args.c,
                    "episode_reward_mean": stop_reward#args.stop_reward,
                    }
    
    def train(self):
        """
        Train an RLlib IMPALA agent using tune until any of the configured stopping criteria is met.
            See https://docs.ray.io/en/latest/tune/api_docs/execution.html#tune-run
        :return: Return the path to the saved agent (checkpoint) and tune's ExperimentAnalysis object
            See https://docs.ray.io/en/latest/tune/api_docs/analysis.html#experimentanalysis-tune-experimentanalysis
        """
        analysis = ray.tune.run(ppo.PPOTrainer, config=self.config_train, local_dir=self.save_dir, stop=self.stop_criteria,
                                checkpoint_at_end=True)
        # list of lists: one list per checkpoint; each checkpoint list contains 1st the path, 2nd the metric value
        checkpoints = analysis.get_trial_checkpoints_paths(trial=analysis.get_best_trial('episode_reward_mean', mode = 'max'),
                                                           metric='episode_reward_mean')
        # retriev the checkpoint path; we only have a single checkpoint, so take the first one
        checkpoint_path = checkpoints[0][0]
        print("Checkpoint path:", checkpoint_path)
        return checkpoint_path, analysis

    def load(self, path):
        """
        Load a trained RLlib agent from the specified path. Call this before testing a trained agent.
        :param path: Path pointing to the agent's saved checkpoint (only used for RLlib agents)
        """
        self.agent = ppo.PPOTrainer(config=self.config)
        self.agent.restore(path)

    def test(self,path):

        """Test trained agent for a single episode. Return the episode reward"""
        # instantiate env class
        unused_shared = []
        unused_own = []
        unsatisfied_shared = []
        unsatisfied_own = []

        episode_reward = 0
        self.config_test["num_workers"] = 0
        self.agent = ppo.PPOTrainer(config=self.config_test)
        self.agent.restore(path)
        env = self.agent.workers.local_worker().env
     
        obs = env.reset()
        done = False

        while not done:
            action = self.agent.compute_action(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward

            unused_shared.append(info["unused_shared"])
            unused_own.append(info["unused_own"])
            unsatisfied_shared.append(info["unsatisfied_shared"])
            unsatisfied_own.append(info["unsatisfied_own"])

        return episode_reward, unused_shared, unused_own, unsatisfied_shared, unsatisfied_own



"""
args = parser.parse_args()
# Class instance
exper = customExperimentClass(3, 1, [8,8,8,4]) # ttl_var, cpt, variable
# Train and save for 2 iterations
checkpoint_path, results = exper.train()
# Load saved
#exper.load(checkpoint_path)
# Test loaded
reward, unused_shared ,unused_own, unsatisfied_shared, unsatisfied_own  = exper.test(checkpoint_path)

print(" info[unused_shared] = ", unused_shared )
print(" info[unused_own] = ", unused_own )
print(" info[unsatisfied_shared] = ", unsatisfied_shared )
print(" info[unsatisfied_own] = ", unsatisfied_own )
print(" reward = ", reward )
"""

