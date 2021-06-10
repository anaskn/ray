import gym
import argparse
import numpy as np

from ray.rllib.env.multi_agent_env import MultiAgentEnv


import ray
from ray import tune
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.models import ModelCatalog
from ray.tune import run_experiments
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer

from multi_agent_env import caching_vM

import pickle
import random


"""

class object
	const
	fct train 
	fct test
	fct plot

"""



def ret_lst(cpt):
    string1 =  'data6/listfile_dist10_'+str(cpt)+'.data' #_evol'+ , _pos'+   #'data4/listfile_40_'+str(cpt)+'.data'
    with open(string1, 'rb') as filehandle:
    # read the data as binary data stream
        lst = pickle.load(filehandle)
    return lst

def ret_nei(cpt):
    string2 = 'data6/nei_tab_pos_dist10_'+str(cpt)+'.data'   #'data4/nei_tab_pos_40_'+str(cpt)+'.data'
    with open(string2, 'rb') as filehandle:
        # read the data as binary data stream
        nei_tab = pickle.load(filehandle)
    return nei_tab

class customExperimentClass():

    def __init__(self,ttl_var=3, cpt=1, variable=[8,8,8,8], stop_iters=5, fcnet_hidd_lst =[[64, 64, 64]],\
                                     fcnet_act_lst =  ["swish", "relu"],lr_lst = [1e-2], stop_timesteps=990000000, stop_reward=0.00001,num_gpus=0, num_gpus_per_worker=0, num_workers=0):

        #fcnet_hidd_lst =[[64, 64, 64]]
        #fcnet_act_lst =  ["relu", "sigmoid"]
        #lr_lst = [1e-2]
        

        self.cpt = cpt
        self.config_train = {
		                    "env": "caching_vM",
		                    
		                    "ttl_var": 3,#ttl_var,
		                    "variable": [8,8,8],#variable,
		                    "nei_tab": ret_nei(cpt),
		                    "lst_tab": ret_lst(cpt),
		              



		                    "seed" : 0,
        }

        
        self.config_test = {
		                    "env": "caching_vM",
		                    
		                    "ttl_var": 3,#ttl_var,
		                    "variable": [8,8,8],#variable,
		                    "nei_tab": ret_nei(cpt),
		                    "lst_tab": ret_lst(cpt),
		                     

		                    "seed" : 0,
        }
        self.save_dir = "~/ray_results"
        self.stop_criteria = {
                    "training_iteration": stop_iters,#args.stop_iters,
                    "timesteps_total": stop_timesteps,#args.c,
                    "episode_reward_mean": stop_reward#args.stop_reward,
                    }
	def env_creator(_):
	        return IrrigationEnv()
	    register_env(env_name, env_creator)
	def gen_policy():
	        return (None, obs_space, act_space, {})
	def policy_mapping_fn(agent_id):
	        return 'agent-' + str(agent_id)

    def train(self, algo):
        """
        Train an RLlib IMPALA agent using tune until any of the configured stopping criteria is met.
            See https://docs.ray.io/en/latest/tune/api_docs/execution.html#tune-run
        :return: Return the path to the saved agent (checkpoint) and tune's ExperimentAnalysis object
            See https://docs.ray.io/en/latest/tune/api_docs/analysis.html#experimentanalysis-tune-experimentanalysis
        """
		# Get environment obs, action spaces and number of agents
	    single_env = caching_vM(config_model)
        env_name = "caching_vM"
	    obs_space = single_env.observation_space
	    act_space = single_env.action_space
	    num_agents = single_env.num_agents

	    # Create a policy mapping
	    

	    policy_graphs = {}
	    for i in range(num_agents):
	        policy_graphs['agent-' + str(i)] = gen_policy()

	    config={        

	                "env": "caching_vM",
	                
	                "env_config": {
	                "ttl_var": self.ttl_var,
	                "variable": self.variable,
	                "nei_tab": ret_nei(self.cpt),
	                "lst_tab": ret_lst(self.cpt),
	                },
	            
	               
	                "model": {
	                    "fcnet_hiddens": [64,64,64],
	                    "fcnet_activation": "relu",
	                    "vf_share_layers": False,#True,
	                },
	                "framework": "torch",# if args.torch else "tf",
	                "num_workers": 3, # parallelism
	                #"seed" : 0,

	                "log_level": "WARN",
	               
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
	                    }

	    # Define experiment details

	    # Initialize ray and run
	    ray.init()
	    #tune.run(**exp_dict)
	    save_dir = "~/ray_results"
	    stop_criteria = {
	            "training_iteration": 2,#stop_iters,#args.stop_iters,
	            }

	    tune.run("PPO",name="my_exp", config=config,   stop=stop_criteria,
	                               )

        select_env = "caching-v020"
        register_env(select_env, lambda config: Caching_v020(self.config_train["env_config"]))

        if algo == "ppo":
            analysis = ray.tune.run(ppo.PPOTrainer, config=self.config_train,  local_dir=self.save_dir, stop=self.stop_criteria,
                               checkpoint_at_end=True)
        if algo == "impala":
            analysis = ray.tune.run(impala.ImpalaTrainer, config=self.config_train, local_dir=self.save_dir, stop=self.stop_criteria,
                                checkpoint_at_end=True)
        if algo == "a3c":
            analysis = ray.tune.run(a3c.A3CTrainer, config=self.config_train, local_dir=self.save_dir, stop=self.stop_criteria,
                                checkpoint_at_end=True)
        if algo == "appo":
            analysis = ray.tune.run(ppo.APPOTrainer, config=self.config_train, local_dir=self.save_dir, stop=self.stop_criteria,
                                checkpoint_at_end=True)
        if algo == "ddpg":
            analysis = ray.tune.run(ddpg.DDPGTrainer, config=self.config_train, local_dir=self.save_dir, stop=self.stop_criteria,
                                checkpoint_at_end=True)
        if algo == "td3":
            analysis = ray.tune.run(ddpg.TD3Trainer, config=self.config_train, local_dir=self.save_dir, stop=self.stop_criteria,
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
        print("lr = ", lr, " fc hid = ", fc_hid, " fc_act = ", fc_act)
        return checkpoint_path, analysis, lr, fc_hid, fc_act

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default= 1)#50)
    parser.add_argument("--stop-timesteps", type=int, default=90000000)
    parser.add_argument("--stop-reward", type=float, default=0.001)
    parser.add_argument("--ttl_var", type=float, default=3)
    parser.add_argument("--cpt", type=float, default=1)
    parser.add_argument("--run", type=str, default="ppo") 
    parser.add_argument("--num_gpus_per_worker", type=float, default= 0)
    parser.add_argument("--num_workers", type=int, default= 0)
    parser.add_argument("--activation", nargs="+", default= ["relu"])
    parser.add_argument('-l','--layer', type=int, nargs='+', required=True, action='append', help='layer list')
    parser.add_argument("--lr", type=float, nargs="+", default=[1e-2])
    parser.add_argument("--gpu", type=float, default= 0)
    parser.add_argument("--cpu", type=int, default= 8)

    ray.shutdown()
    ray.init()

    args = parser.parse_args()

    exper = customExperimentClass(args.ttl_var, args.cpt, [8,8,4], \
            fcnet_hidd_lst = args.layer, fcnet_act_lst = args.activation, lr_lst = args.lr, stop_iters=args.epochs, num_gpus=args.gpu, num_gpus_per_worker=args.num_gpus_per_worker, num_workers=args.num_workers)                                  
        