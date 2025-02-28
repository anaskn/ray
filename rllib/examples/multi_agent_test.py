import gym
import numpy as np

from ray.rllib.env.multi_agent_env import MultiAgentEnv


import ray
from ray import tune
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.models import ModelCatalog
from ray.tune import run_experiments
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer



class IrrigationEnv(MultiAgentEnv):
    def __init__(self, return_agent_actions = False, part=False):
        self.num_agents = 5
        self.observation_space = gym.spaces.Box(low=0, high=800, shape=(1,))
        self.action_space = gym.spaces.Box(low=0, high=5, shape=(1,))

    def reset(self):
        obs = {}
        self.dones = set()
        self.water = np.random.uniform(200,800)
        for i in range(self.num_agents):
            obs[i] = np.array([self.water])
            #print("obs[i] = ", obs[i])
        #print("obs total == ", obs )
        return obs

    def cal_rewards(self, action_dict):
        self.curr_water = self.water
        reward = 0
        for i in range(self.num_agents):
            water_demanded = self.water*action_dict[i][0]

            if self.curr_water == 0:
                reward += 0
                reward -= water_demanded*100
            elif self.curr_water - water_demanded<0:
                water_needed = water_demanded - self.curr_water
                water_withdrawn = self.curr_water
                self.curr_water = 0
                reward += -water_withdrawn**2 + 200*water_withdrawn
                reward -= water_needed*100
            else:
                self.curr_water -= water_demanded
                water_withdrawn = water_demanded
                reward += -water_withdrawn**2 + 200*water_withdrawn

        return reward

    def step(self, action_dict):
        obs, rew, done, info = {}, {}, {}, {}

        reward = self.cal_rewards(action_dict)

        for i in range(self.num_agents):

            obs[i], rew[i], done[i], info[i] = np.array([self.curr_water]), reward, True, {}

        done["__all__"] = True
        print("done === ", done)
        # print(self.observation_space)
        return obs, rew, done, info

# Driver code for training
def setup_and_train():

    # Create a single environment and register it
    def env_creator(_):
        return IrrigationEnv()
    single_env = IrrigationEnv()
    env_name = "IrrigationEnv"
    register_env(env_name, env_creator)

    # Get environment obs, action spaces and number of agents
    obs_space = single_env.observation_space
    act_space = single_env.action_space
    num_agents = single_env.num_agents

    # Create a policy mapping
    def gen_policy():
        return (None, obs_space, act_space, {})

    policy_graphs = {}
    for i in range(num_agents):
        policy_graphs['agent-' + str(i)] = gen_policy()

    def policy_mapping_fn(agent_id):
        return 'agent-' + str(agent_id)

    # Define configuration with hyperparam and training details
    config={
                "log_level": "WARN",
                "num_workers": 3,
                "num_cpus_for_driver": 1,
                "num_cpus_per_worker": 1,
                "num_sgd_iter": 10,
                "train_batch_size": 128,
                "lr": 5e-3,
                "model":{"fcnet_hiddens": [8, 8]},
                "multiagent": {
                    "policies": policy_graphs,
                    "policy_mapping_fn": policy_mapping_fn,
                },
                "env": "IrrigationEnv"}

    # Define experiment details
    exp_name = 'my_exp'
    exp_dict = {
            'name': exp_name,
            'run_or_experiment': 'PPO',
            "stop": {
                "training_iteration": 100
            },
            'checkpoint_freq': 20,
            "config": config,
        }

    # Initialize ray and run
    ray.init()
    tune.run(**exp_dict)

if __name__=='__main__':
    
    setup_and_train()