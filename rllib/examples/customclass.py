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
    def __init__(self):
        #ray.shutdown()
        #ray.init(num_cpus=2, num_gpus=0)
        self.env = ContentCaching#gym.make("ContentCaching-v0")
        self.config_train = {
                        "env": ContentCaching,
                        "env_config": {
                        "ttl_var": 3,
                        "cpt": 2,
                        "variable": [8,8,8,4],
                        "nei_tab": ret_nei(2),
                        "lst_tab": ret_lst(2),

        }}
        self.config_test = {
                        "env": ContentCaching,
                        "env_config": {
                        "ttl_var": 3,
                        "cpt": 2,
                        "variable": [8,8,8,4],
                        "nei_tab": ret_nei(2),
                        "lst_tab": ret_lst(2),

        }}
        self.save_dir = "~/ray_results"
        self.stop_criteria = {
                    "training_iteration": args.stop_iters,
                    "timesteps_total": args.stop_timesteps,
                    "episode_reward_mean": args.stop_reward,
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
        
        #results = tune.run("IMPALA",
         #           verbose=1,
          #          config=self.config,
           #         stop={"training_iteration":  2,},
            #        checkpoint_freq=1,
             #       keep_checkpoints_num=1,
              #      checkpoint_score_attr='training_iteration',
               #    )


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
        #env = self.env


        episode_reward = 0
        self.config_test["num_workers"] = 0
        #eval_agent = ppo.PPOTrainer(config=self.config_test, env=ContentCaching)#"ContentCaching-v0")
        #eval_agent.restore(path)
        #env = eval_agent.workers.local_worker().env

        self.agent = ppo.PPOTrainer(config=self.config_test)
        self.agent.restore(path)
        env = self.agent.workers.local_worker().env



        obs = env.reset()
        done = False
        while not done:
            action = self.agent.compute_action(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            unused_shared = info["unused_shared"]

        return episode_reward, unused_shared



args = parser.parse_args()
# Class instance
exper = customExperimentClass()
# Train and save for 2 iterations
checkpoint_path, results = exper.train()
# Load saved
#exper.load(checkpoint_path)
# Test loaded
re, unused_hared = exper.test(checkpoint_path)

print(" info[unused_shared] = ", unused_hared )
print(" reward = ", re )
