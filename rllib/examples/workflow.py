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

def train(config, save_dir, stop_criteria):
    """
    Train an RLlib PPO agent using tune until any of the configured stopping criteria is met.
    :param stop_criteria: Dict with stopping criteria.
        See https://docs.ray.io/en/latest/tune/api_docs/execution.html#tune-run
    :return: Return the path to the saved agent (checkpoint) and tune's ExperimentAnalysis object
        See https://docs.ray.io/en/latest/tune/api_docs/analysis.html#experimentanalysis-tune-experimentanalysis
    """
    analysis = ray.tune.run(ppo.PPOTrainer, config=config, local_dir=save_dir, stop=stop_criteria,
                            checkpoint_at_end=True)
    # list of lists: one list per checkpoint; each checkpoint list contains 1st the path, 2nd the metric value
    
    checkpoints = analysis.get_trial_checkpoints_paths(trial=analysis.get_best_trial('episode_reward_mean',mode = 'max'), 
                                                       metric='episode_reward_mean')

    print("okokkokokokokokokokokokokkkkkkkkkkkkkkkkkkkooooooooooooooooooooookkkkkkkkkkkkkkkk")
    # retriev the checkpoint path; we only have a single checkpoint, so take the first one
    checkpoint_path = checkpoints[0][0]
    print("okokkokokokokokokokokokokkkkkkkkkkkkkkkkkkkooooooooooooooooooooookkkkkkkkkkkkkkkk")
    print("Checkpoint path:", checkpoint_path)
    return checkpoint_path, analysis

def load(path):
    """
    Load a trained RLlib agent from the specified path. Call this before testing a trained agent.
    :param path: Path pointing to the agent's saved checkpoint (only used for RLlib agents)
    """
    agent = ppo.PPOTrainer(config=config)#, env=self.env_class)
    agent.restore(path)
    return agent

def test():
    """Test trained agent for a single episode. Return the episode reward"""
    # instantiate env class
    env = ContentCaching #self.env_class(self.env_config)

    # run until episode ends
    episode_reward = 0
    done = False
    obs = env.reset()
    while not done:
        action = self.agent.compute_action(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward

    return episode_reward



if __name__ == "__main__":

    args = parser.parse_args()
    
    #ray.init(num_cpus=3)#num_cpus=3
    #time.sleep(60)

    
    config = {
        "env": ContentCaching,
        "env_config": {
            "ttl_var": 3,
            "cpt": 2,
            "variable": [8,8,8,4],
            "nei_tab": ret_nei(2),
            "lst_tab": ret_lst(2),

        }}
    stop_criteria = {
    "training_iteration": args.stop_iters,
    "timesteps_total": args.stop_timesteps,
    "episode_reward_mean": args.stop_reward,
    }

    save_dir = "~/ray_results"  
    checkpoint_path, analysis = train(config, save_dir, stop_criteria)
    load(checkpoint_path)
    print("okokkokokokokokokokokokokkkkkkkkkkkkkkkkkkkooooooooooooooooooooookkkkkkkkkkkkkkkk")
    test()


