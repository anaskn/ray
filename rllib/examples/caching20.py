
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
import pickle
import time
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt 
from gym_example.envs.caching_env20 import Caching_v020
import random



#import torch
#torch.set_deterministic(True)



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

    #pplot(scores)
    window_width= 10
    cumsum_vec = np.cumsum(np.insert(sum_l, 0, 0))
    sum_l = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width

    plt.plot(sum_l , color='orange', linestyle='dotted', marker='x',label=args.run+'_Reward')

    plt.ylabel('Reward', size= 8 )
    plt.xlabel('$'+args.run+'$', size= 10)

    plt.xticks(size = 7)
    plt.yticks(size = 7)

    # Add a legend
    plt.legend()
    plt.grid()
    
    # save file .pdf
    plt.savefig('plot/Reward_'+args.run+'.pdf')

    our_file = [sum_l]
    with open('model/Reward_'+args.run+'.data', 'wb') as filehandle:   #unused
    #  # store the data as binary data stream
      pickle.dump(our_file, filehandle)
    plt.show()

def ret_lst(cpt):
    string1 =  'data6/listfile_dist10_'+str(cpt)+'.data' #_evol'+ , _pos'+   #'data4/listfile_40_'+str(cpt)+'.data'
    #string1 =  'data6/listfile_dist10_'+str(cpt)+'.data' #_evol'+ , _pos'+   #'data4/listfile_40_'+str(cpt)+'.data'
    with open(string1, 'rb') as filehandle:
    # read the data as binary data stream
        lst = pickle.load(filehandle)
    return lst

def ret_nei(cpt):
    string2 = 'data6/nei_tab_pos_dist10_'+str(cpt)+'.data'   #'data4/nei_tab_pos_40_'+str(cpt)+'.data'
    #string2 = 'data6/nei_tab_pos_dist10_'+str(cpt)+'.data'   #'data4/nei_tab_pos_40_'+str(cpt)+'.data'
    with open(string2, 'rb') as filehandle:
        # read the data as binary data stream
        nei_tab = pickle.load(filehandle)
    return nei_tab

class customExperimentClass():

    def __init__(self,algo,ttl_var=3, cpt=1, variable=[8,8,8,8], stop_iters=5, fcnet_hidd_lst =[[64, 64, 64]],\
                                     fcnet_act_lst =  ["swish", "relu"],lr_lst = [1e-9], stop_timesteps=990000000, stop_reward=99999999,num_gpus=0, num_gpus_per_worker=0, num_workers=0):

        #fcnet_hidd_lst =[[64, 64, 64]]
        #fcnet_act_lst =  ["relu", "sigmoid"]
        #lr_lst = [1e-2]
        

        if algo == 'ppo' or algo== 'appo':
            self.config_train = {
                            "env": "caching-v020",
                            "env_config": {
                            "ttl_var": ttl_var,
                            "variable": variable,
                            "nei_tab": ret_nei(cpt),
                            "lst_tab": ret_lst(cpt),
                            },
                            "num_gpus": num_gpus,
                            "num_gpus_per_worker": num_gpus_per_worker,
                            "model": {
                                "fcnet_hiddens": grid_search( fcnet_hidd_lst),
                                "fcnet_activation": grid_search(fcnet_act_lst),
                                "vf_share_layers": False,#True,
                                
                            },
                            "framework": grid_search(["torch"]),# if args.torch else "tf",
                            "lr": grid_search(lr_lst),  # try different lrs
                            "num_workers": num_workers, # parallelism

                            "seed" : 0,
                            "vf_loss_coeff" : grid_search([0.5]),
                            "kl_target" : grid_search([0.003]),
                            "clip_param" : grid_search([0.2]),

            }
        
            self.config_test = {
                            "env": "caching-v020",
                            "env_config": {
                            "ttl_var": ttl_var,
                            "variable": variable,
                            "nei_tab": ret_nei(11),#11
                            "lst_tab": ret_lst(11), #11                       

                            },
                            "model": {
                                "fcnet_hiddens": [64, 64, 64],
                                "fcnet_activation": "relu",
                                "vf_share_layers": False,#True,
                            },
                            "num_gpus": num_gpus,
                            "num_gpus_per_worker": num_gpus_per_worker,
                            "num_workers": num_workers, # parallelism
                            "lr": [1e-2],  # try different lrs
                            "num_workers": 2,  # parallelism
                            "framework": "torch",# if args.torch else "tf",
                            "seed" : 0,
                            "vf_loss_coeff" : 1,
                            "kl_target" : 0.03,
                            "clip_param" : 0.2,
            }

        if algo == 'ddpg' or algo == 'td3':
            self.config_train = {
                            "env": "caching-v020",
                            "env_config": {
                            "ttl_var": ttl_var,
                            "variable": variable,
                            "nei_tab": ret_nei(cpt),
                            "lst_tab": ret_lst(cpt),
                            },
                            "num_gpus": num_gpus,
                            "num_gpus_per_worker": num_gpus_per_worker,
                            "model": {
                                "fcnet_hiddens": grid_search( fcnet_hidd_lst),
                                "fcnet_activation": grid_search(fcnet_act_lst),
                                "vf_share_layers": False,#True,
                                
                            },
                            "framework": grid_search(["torch"]),# if args.torch else "tf",
                            "lr": grid_search(lr_lst),  # try different lrs
                            "num_workers": num_workers, # parallelism

                            "seed" : 0,
                            "target_noise_clip" : grid_search([0.1]),
                            "learning_starts" : grid_search([800]),


            }
        
            self.config_test = {
                            "env": "caching-v020",
                            "env_config": {
                            "ttl_var": ttl_var,
                            "variable": variable,
                            "nei_tab": ret_nei(11),#11
                            "lst_tab": ret_lst(11), #11                       

                            },
                            "model": {
                                "fcnet_hiddens": [64, 64, 64],
                                "fcnet_activation": "relu",
                                "vf_share_layers": False,#True,
                            },
                            "num_gpus": num_gpus,
                            "num_gpus_per_worker": num_gpus_per_worker,
                            "num_workers": num_workers, # parallelism
                            "lr": [1e-2],  # try different lrs
                            "num_workers": 2,  # parallelism
                            "framework": "torch",# if args.torch else "tf",
                            "seed" : 0,
                            "target_noise_clip" : 0.1,
                            "learning_starts" : 800,
            }

        if algo == 'a2c' or algo == 'a3c':
            self.config_train = {
                            "env": "caching-v020",
                            "env_config": {
                            "ttl_var": ttl_var,
                            "variable": variable,
                            "nei_tab": ret_nei(cpt),
                            "lst_tab": ret_lst(cpt),
                            },
                            "num_gpus": num_gpus,
                            "num_gpus_per_worker": num_gpus_per_worker,
                            "model": {
                                "fcnet_hiddens": grid_search( fcnet_hidd_lst),
                                "fcnet_activation": grid_search(fcnet_act_lst),
                                "vf_share_layers": False,#True,
                                
                            },
                            "framework": grid_search(["torch"]),#"torch",# if args.torch else "tf",
                            "lr": grid_search(lr_lst),  # try different lrs
                            "num_workers": 1,#num_workers, # parallelism

                            "seed" : 0,
                            "vf_loss_coeff" : grid_search([0.7]),
                            "grad_clip" : grid_search([40]),


            }
        
            self.config_test = {
                            "env": "caching-v020",
                            "env_config": {
                            "ttl_var": ttl_var,
                            "variable": variable,
                            "nei_tab": ret_nei(11),#11
                            "lst_tab": ret_lst(11), #11                       

                            },
                            "model": {
                                "fcnet_hiddens": [64, 64, 64],
                                "fcnet_activation": "relu",
                                "vf_share_layers": False,#True,
                            },
                            "num_gpus": num_gpus,
                            "num_gpus_per_worker": num_gpus_per_worker,
                            "num_workers": num_workers, # parallelism
                            "lr": [1e-3],  # try different lrs
                            "num_workers": 1,  # parallelism
                            "framework": "torch",#"torch",# if args.torch else "tf",
                            "seed" : 0,
                            "vf_loss_coeff" : 0.5,
                            "grad_clip" : 30,
            }

        self.save_dir = "~/ray_results"
        self.stop_criteria = {
                    "training_iteration": stop_iters,#args.stop_iters,
                    "timesteps_total": 900000,#stop_timesteps,#args.c,
                    "episode_reward_mean": 900000,#stop_reward#args.stop_reward,
                    #stop_timesteps=990000000, stop_reward=99999999
                    }


    def train(self, algo):
        """
        Train an RLlib IMPALA agent using tune until any of the configured stopping criteria is met.
            See https://docs.ray.io/en/latest/tune/api_docs/execution.html#tune-run
        :return: Return the path to the saved agent (checkpoint) and tune's ExperimentAnalysis object
            See https://docs.ray.io/en/latest/tune/api_docs/analysis.html#experimentanalysis-tune-experimentanalysis
        """

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
        frame_work = analysis.get_best_config(metric='episode_reward_mean', mode="max")["framework"] 
        if algo=='ppo' or algo== 'appo':
            vf_loss = analysis.get_best_config(metric='episode_reward_mean', mode="max")["vf_loss_coeff"] 
            kl_tar = analysis.get_best_config(metric='episode_reward_mean', mode="max")["kl_target"] 
            clip = analysis.get_best_config(metric='episode_reward_mean', mode="max")["clip_param"]
        if algo =="ddpg" or algo == "td3":
            target_noise = analysis.get_best_config(metric='episode_reward_mean', mode="max")["target_noise_clip"] 
            lea_starts = analysis.get_best_config(metric='episode_reward_mean', mode="max")["learning_starts"]
        if algo =="a2c" or algo == "a3c":
            vf_loss = analysis.get_best_config(metric='episode_reward_mean', mode="max")["vf_loss_coeff"] 
            grad = analysis.get_best_config(metric='episode_reward_mean', mode="max")["grad_clip"]


        # list of lists: one list per checkpoint; each checkpoint list contains 1st the path, 2nd the metric value
        checkpoints = analysis.get_trial_checkpoints_paths(trial=analysis.get_best_trial('episode_reward_mean', mode = 'max'),
                                                           metric='episode_reward_mean')
        # retriev the checkpoint path; we only have a single checkpoint, so take the first one
        print("checkpoints path:", len(checkpoints))
        checkpoint_path = checkpoints[0][0]
        print("Checkpoint path:", checkpoint_path)

        print("lr = ", lr, " fc hid = ", fc_hid, " fc_act = ", fc_act)

        if algo =="ppo" or algo== 'appo':
            return checkpoint_path, analysis, lr, fc_hid, frame_work, fc_act, vf_loss, kl_tar, clip
        if algo =="ddpg" or algo == "td3":
            return checkpoint_path, analysis, lr, fc_hid, frame_work, fc_act, target_noise, lea_starts
        if algo =="a2c" or algo == "a3c":
            return checkpoint_path, analysis, lr, fc_hid,frame_work, fc_act, vf_loss, grad


    def test(self,algo, path, lr, fc_hid, frame_work, fc_act, vf_loss, kl_target, clip,  target_noise, lea_starts, grad):

        """Test trained agent for a single episode. Return the episode reward"""
        # instantiate env class
        unused_shared = []
        unused_own = []
        unsatisfied_shared = []
        unsatisfied_own = []

        episode_reward = 0
        
        self.config_test["num_workers"] = 1
        self.config_test["lr"] = lr
        self.config_test['model']["fcnet_hiddens"] = fc_hid
        self.config_test['model']["fcnet_activation"] = fc_act
        self.config_test["framework"] = frame_work

        if algo == 'ppo' or algo == 'appo':
            self.config_test["vf_loss_coeff"] = vf_loss
            self.config_test["kl_target"] = kl_target
            self.config_test["clip_param"] = clip
        if algo == 'ddpg' or algo == 'td3':
            self.config_test["target_noise_clip"] = target_noise
            self.config_test["learning_starts"] = lea_starts
        if algo == 'a2c' or algo == 'a3c':
            self.config_test["vf_loss_coeff"] = vf_loss
            self.config_test["grad_clip"] = grad
        

        if algo == "ppo":
            self.agent = ppo.PPOTrainer(config=self.config_test)
        if algo == "ddpg":
            self.agent = ddpg.DDPGTrainer(config=self.config_test)
        if algo == "a3c":
            self.agent = a3c.A3CTrainer(config=self.config_test)
        if algo == "impala":
            self.agent = impala.ImpalaTrainer(config=self.config_test)
        if algo == "appo":
            self.agent = ppo.APPOTrainer(config=self.config_test)
        if algo == "td3":
            self.agent = ddpg.TD3Trainer(config=self.config_test)
        
        self.agent.restore(path)

        env = gym.make("caching-v020", config=self.config_test["env_config"])
        
     
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


    def test_random(self,algo):

        """Test trained agent for a single episode. Return the episode reward"""
        # instantiate env class
        unused_shared = []
        unused_own = []
        unsatisfied_shared = []
        unsatisfied_own = []

        episode_reward = 0
        

        env = gym.make("caching-v020", config=self.config_test["env_config"])
        
     
        obs = env.reset()
        done = False

        while not done:
            action = [random.uniform(0, 1) for x in range(20)]#self.agent.compute_action(obs)
            print("len action random  ======== ", len(action))
            obs, reward, done, info = env.step(action)
            episode_reward += reward

            unused_shared.append(info["unused_shared"])
            unused_own.append(info["unused_own"])
            unsatisfied_shared.append(info["unsatisfied_shared"])
            unsatisfied_own.append(info["unsatisfied_own"])
        
        return episode_reward, unused_shared, unused_own, unsatisfied_shared, unsatisfied_own



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
    parser.add_argument("--lr", type=float, nargs="+", default=[1e-6])
    parser.add_argument("--gpu", type=float, default= 0)
    parser.add_argument("--cpu", type=int, default= 8)


    args = parser.parse_args()


    ray.shutdown()
    ray.init()


    exper = customExperimentClass(args.ttl_var, args.cpt, [8,8,4], \
            fcnet_hidd_lst = args.layer, fcnet_act_lst = args.activation, lr_lst = args.lr, stop_iters=args.epochs, num_gpus=args.gpu, num_gpus_per_worker=args.num_gpus_per_worker, num_workers=args.num_workers)                                  
        
    #checkpoint_path, results, lr, fc_hid, fc_act = exper.train(args.run)
    all_in = exper.train(args.run)
    checkpoint_path=all_in[0]
    results= all_in[1]
    lr= all_in[2]
    fc_hid= all_in[3]
    frame_work = all_in[4]
    fc_act= all_in[5]


    if args.run== 'ppo' or args.run== 'appo':
        vf_loss=all_in[6]
        kl_target= all_in[7]
        clip=all_in[8]

    if args.run== 'ddpg' or args.run== 'td3':
        target_noise=all_in[6]
        lea_starts=all_in[7]
    if args.run== 'a2c' or args.run== 'a3c':
        vf_loss=all_in[6]
        grad=all_in[7]
        



    #the_plot(results)
    print("gym.make successfully")

    if args.run== 'ppo'  or args.run== 'appo':
        reward, unused_shared ,unused_own, unsatisfied_shared, unsatisfied_own  = exper.test(args.run ,checkpoint_path, lr, fc_hid, frame_work, fc_act, vf_loss, kl_target, clip, -1, -1, -1)
    if args.run== 'ddpg' or args.run== 'td3':
        reward, unused_shared ,unused_own, unsatisfied_shared, unsatisfied_own  = exper.test(args.run ,checkpoint_path, lr, fc_hid, frame_work, fc_act, -1,         -1,      -1,   target_noise, lea_starts, -1)
    if args.run== 'a2c' or args.run== 'a3c':
        reward, unused_shared ,unused_own, unsatisfied_shared, unsatisfied_own  = exper.test(args.run ,checkpoint_path, lr, fc_hid, frame_work, fc_act, vf_loss,    -1,      -1,   -1, -1,  grad)

    #print(" info[unused_shared] = ", unused_shared )
    #print(" info[unused_own] = ", unused_own )
    #print(" info[unsatisfied_shared] = ", unsatisfied_shared )
    #print(" info[unsatisfied_own] = ", unsatisfied_own )
    #print(" reward = ", reward )
    
    print(" best lr = ", lr)
    print(" best hidden layer parameter = ", fc_hid)
    print(" best fonction activation  = ", fc_act)
    print("framework ", frame_work)


    if args.run== 'ppo' or args.run== 'appo':

        print("best vf_loss =", vf_loss)
        print("kl_target", kl_target)
        print("clip ", clip)
    if args.run== 'ddpg' or args.run== 'td3':

        print("target_noise ", target_noise)
        print("lea_starts ", lea_starts)

    if args.run== 'a2c' or args.run== 'a3c':

        print("best vf_loss =", vf_loss)
        print("clip grad ", grad)
    
