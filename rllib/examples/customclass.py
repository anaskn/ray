import argparse

import ray
from ray import tune
from ray.rllib.agents import ppo
from ray.rllib.agents import ddpg
from ray.rllib.agents import a3c



from ray.tune import grid_search
from my_env import ContentCaching
import pickle
import time
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt 



def the_plot(analysis):


    dfs = analysis.trial_dataframes
    # Plot by epoch
    ax = None  # This plots everything on the same plot
    for d in dfs.values():
        ax = d["hist_stats/episode_reward"]

    sum_l =[]
    for x in range(len(ax)):
        res = ax[x].strip('][').split(', ')
        l1= [float(x) for x in res]
        sum_l.extend(l1)

    print("len sum_l ===== : ", len(sum_l))

    plt.plot(sum_l , color='orange', linestyle='dotted', marker='x',label=args.algo+'_Reward')

    plt.ylabel('Reward', size= 8 )
    plt.xlabel('$'+args.algo+'$', size= 10)

    plt.xticks(size = 7)
    plt.yticks(size = 7)

    # Add a legend
    plt.legend()
    
    # save file .pdf
    plt.savefig('plot/Reward_'+args.algo+'.pdf') #relusigmoid
    plt.show()

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

    def __init__(self,ttl_var, cpt, variable, stop_iters=2, fcnet_hidd_lst =[[64, 64, 64]],\
                                     fcnet_act_lst =  ["relu", "sigmoid"],lr_lst = [1e-2], stop_timesteps=990000000, stop_reward=0.00001):#

        #fcnet_hidd_lst =[[64, 64, 64]]
        #fcnet_act_lst =  ["relu", "sigmoid"]
        #lr_lst = [1e-2]
        self.env = ContentCaching#gym.make("ContentCaching-v0")
        self.config_train = {
                        "env": ContentCaching,
                        "env_config": {
                        "ttl_var": ttl_var,
                        "variable": variable,#[8,8,8,4],
                        "nei_tab": ret_nei(cpt),
                        "lst_tab": ret_lst(cpt),
                        },
                        #"num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),

                        "model": {
                            # By default, the MODEL_DEFAULTS dict above will be used.

                            # Change individual keys in that dict by overriding them, e.g.
                            
                            "fcnet_hiddens": grid_search( fcnet_hidd_lst),
                            "fcnet_activation": grid_search(fcnet_act_lst),
                            "vf_share_layers": False,#True,
                        },
                        
                        "lr": grid_search(lr_lst),  # try different lrs
                        "num_workers": 4,  # parallelism
                        "seed" : 0
                        #"framework": "torch" if args.torch else "tf",
        }

        
        self.config_test = {
                        "env": ContentCaching,
                        "env_config": {
                        "ttl_var": ttl_var,
                        "variable": variable,
                        "nei_tab": ret_nei(5),
                        "lst_tab": ret_lst(5),                        

                        },
                        "model": {
                            # By default, the MODEL_DEFAULTS dict above will be used.

                            # Change individual keys in that dict by overriding them, e.g.
                            "fcnet_hiddens": [64, 64, 64],
                            "fcnet_activation": "sigmoid",
                            "vf_share_layers": False,#True,
                        },

                        "lr": [1e-2],  # try different lrs
                        #"num_workers": 2,  # parallelism
                        #"framework": "torch" if args.torch else "tf",
        }
        self.save_dir = "~/ray_results"
        self.stop_criteria = {
                    "training_iteration": stop_iters,#args.stop_iters,
                    "timesteps_total": stop_timesteps,#args.c,
                    "episode_reward_mean": stop_reward#args.stop_reward,
                    }
    
    def train(self, algo):
        """
        Train an RLlib IMPALA agent using tune until any of the configured stopping criteria is met.
            See https://docs.ray.io/en/latest/tune/api_docs/execution.html#tune-run
        :return: Return the path to the saved agent (checkpoint) and tune's ExperimentAnalysis object
            See https://docs.ray.io/en/latest/tune/api_docs/analysis.html#experimentanalysis-tune-experimentanalysis
        """
        if algo == "ppo":
            analysis = ray.tune.run(ppo.PPOTrainer, config=self.config_train, local_dir=self.save_dir, stop=self.stop_criteria,
                               checkpoint_at_end=True)
        if algo == "ddpg":
            analysis = ray.tune.run(ddpg.DDPGTrainer, config=self.config_train, local_dir=self.save_dir, stop=self.stop_criteria,
                                checkpoint_at_end=True)
        if algo == "a3c":
            analysis = ray.tune.run(a3c.A3CTrainer, config=self.config_train, local_dir=self.save_dir, stop=self.stop_criteria,
                                checkpoint_at_end=True)
        if algo == "td3":
            analysis = ray.tune.run(ddpg.TD3Trainer, config=self.config_train, local_dir=self.save_dir, stop=self.stop_criteria,
                                checkpoint_at_end=True)
        if algo == "appo":
            analysis = ray.tune.run(ppo.APPOTrainer, config=self.config_train, local_dir=self.save_dir, stop=self.stop_criteria,
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

    def load(self, path):
        """
        Load a trained RLlib agent from the specified path. Call this before testing a trained agent.
        :param path: Path pointing to the agent's saved checkpoint (only used for RLlib agents)
        """
        self.agent = ppo.PPOTrainer(config=self.config)
        self.agent.restore(path)



    def test(self,algo, path, lr, fc_hid, fc_act):

        """Test trained agent for a single episode. Return the episode reward"""
        # instantiate env class
        unused_shared = []
        unused_own = []
        unsatisfied_shared = []
        unsatisfied_own = []

        episode_reward = 0
        self.config_test["num_workers"] = 0
        self.config_test["lr"] = lr
        self.config_test['model']["fcnet_hiddens"] = fc_hid
        self.config_test['model']["fcnet_activation"] = fc_act

        if algo == "ppo":
            self.agent = ppo.PPOTrainer(config=self.config_test)
        if algo == "ddpg":
            self.agent = ddpg.DDPGTrainer(config=self.config_test)
        if algo == "a3c":
            self.agent = a3c.A3CTrainer(config=self.config_test)
        if algo == "td3":
            self.agent = ddpg.TD3Trainer(config=self.config_test)
        if algo == "appo":
            self.agent = ppo.APPOTrainer(config=self.config_test)

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



if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default= 3)#50)
    parser.add_argument("--stop-timesteps", type=int, default=90000000)
    parser.add_argument("--stop-reward", type=float, default=0.001)
    parser.add_argument("--ttl_var", type=float, default=3)
    parser.add_argument("--cpt", type=float, default=1)
    parser.add_argument("--algo", type=str, default="ppo")   


    ray.shutdown()
    ray.init(num_cpus=3)#num_cpus=2, num_gpus=0)

    args = parser.parse_args()
    # Class instance

    #print("num gpu = ",int(os.environ.get("RLLIB_NUM_GPUS", "1")) )
    """
    exper = customExperimentClass(args.ttl_var, args.cpt, [8,8,8,4], args.epochs) # ttl_var, cpt, variable

    # Train and save for 2 iterations
    checkpoint_path, results, lr, fc_hid, fc_act = exper.train(args.algo)
    the_plot(results)
    
    print("------------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------------")
    
    # Load saved
    #exper.load(checkpoint_path)
    # Test loaded
    """
    """
    reward, unused_shared ,unused_own, unsatisfied_shared, unsatisfied_own  = exper.test(args.algo,checkpoint_path, lr, fc_hid, fc_act)
   
    print(" info[unused_shared] = ", unused_shared )
    print(" info[unused_own] = ", unused_own )
    print(" info[unsatisfied_shared] = ", unsatisfied_shared )
    print(" info[unsatisfied_own] = ", unsatisfied_own )
    print(" reward = ", reward )
    """
    
    

"""

 

    config=dict(
        extra_config,
        **{
            "env": "BreakoutNoFrameskip-v4"
            if args.use_vision_network else "CartPole-v0",
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "callbacks": {
                "on_train_result": check_has_custom_metric,
            },
            "model": {
                "custom_model": "keras_q_model"
                if args.run == "DQN" else "keras_model"
            },
            "framework": "tf",
        })

"""