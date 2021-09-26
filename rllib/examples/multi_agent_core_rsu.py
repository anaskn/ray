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
from ray.rllib.agents import ddpg
from ray.rllib.agents import a3c
import ray.rllib.agents.impala as impala

from multi_agent_env_rsu import caching_vM 

from ray.rllib.agents import ppo

import pickle
import random
import ast
from ray.tune import grid_search
import matplotlib.pyplot as plt 

#import torch
#torch.set_deterministic(True)


import os
import sys

os.environ.setdefault("TUNE_GLOBAL_CHECKPOINT_S", str(sys.maxsize))

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
    plt.savefig('plot/Reward_multi_agent_RSU'+args.run+'.pdf')
    #plt.show()
    our_file = [sum_l]
    with open('model8/Reward_multi_agent_RSU'+args.run+'.data', 'wb') as filehandle:   #unused
    #  # store the data as binary data stream
      pickle.dump(our_file, filehandle)


def ret_lst(cpt):
    string1 =  'data8/listfile_dist10_'+str(cpt)+'.data' #it was 8 #_evol'+ , _pos'+   #'data4/listfile_40_'+str(cpt)+'.data' 
    with open(string1, 'rb') as filehandle:
    # read the data as binary data stream
        lst = pickle.load(filehandle)
    return lst

def ret_nei(cpt):
    string2 = 'data8/nei_tab_pos_dist10_'+str(cpt)+'.data'   #it was 8  #'data4/nei_tab_pos_40_'+str(cpt)+'.data'
    with open(string2, 'rb') as filehandle:
        # read the data as binary data stream
        nei_tab = pickle.load(filehandle)
    return nei_tab

class customExperimentClass():

    def __init__(self,num_agents,algo, ttl_var=3, cpt=1, variable=[8,8,8,8], stop_iters=1, fcnet_hidd_lst =[[64, 64, 64]],\
                                     fcnet_act_lst =  ["swish"],lr_lst = [5e-3], stop_timesteps=999990000, stop_reward=99999999,num_gpus=0, num_gpus_per_worker=0, num_workers=0):

        #Get environment obs, action spaces and number of agents
        def gen_policy():
           return (None, obs_space, act_space, {})
        def policy_mapping_fn(agent_id):
           return 'agent-' + str(agent_id)

        env_name = "caching_vM"


        obs_space = gym.spaces.Box(low=0, high=200, shape=(3,), dtype=np.float32)#single_env.observation_space
        act_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)#single_env.action_space
        #num_agents = 22#single_env.num_agents

        # Create a policy mapping
        policy_graphs = {}
        for i in range(num_agents):
            policy_graphs['agent-' + str(i)] = gen_policy()
        
        self.cpt = cpt
        if algo == 'ppo' or algo == 'appo':
          self.config={        

                     "env": "caching_vM",
                     "env_config": {
                     "ttl_var": ttl_var,
                     "variable": variable,
                     "num_agents": num_agents,
                     "nei_tab": ret_nei(cpt),
                     "lst_tab": ret_lst(cpt),
                     },

                     "model": {
                         "fcnet_hiddens": grid_search(fcnet_hidd_lst),#[64,64,64],
                         "fcnet_activation": grid_search(fcnet_act_lst),#"relu",
                         "vf_share_layers": False,#True,
                     },
                     "framework": "torch",# if args.torch else "tf",
                     "num_workers": num_workers, # parallelism
                     "num_gpus":num_gpus,
                     "num_gpus_per_worker":num_gpus_per_worker,
                     "seed" : 0,
                     "vf_loss_coeff" : 1,#grid_search([0.5,1]),
                     "kl_target" : 0.03,#grid_search([0.003,0.03]),
                     "clip_param" : 0.2,#grid_search([0.1,0.2,0.3]),

                     "log_level": "WARN",
                    
                     "num_cpus_for_driver": 1,
                     "num_cpus_per_worker": 1,

                     "train_batch_size": 128,
                     "lr": grid_search(lr_lst),
                     "multiagent": {
                         "policies": policy_graphs,
                         "policy_mapping_fn": policy_mapping_fn,
                     },
                         }
        if algo == 'ddpg' or algo == 'td3':
          self.config={        

                     "env": "caching_vM",
                     "env_config": {
                     "ttl_var": ttl_var,
                     "variable": variable,
                     "num_agents": num_agents,
                     "nei_tab": ret_nei(cpt),
                     "lst_tab": ret_lst(cpt),
                     },

                     "model": {
                         "fcnet_hiddens": grid_search(fcnet_hidd_lst),#[64,64,64],
                         "fcnet_activation": grid_search(fcnet_act_lst),#"relu",
                         "vf_share_layers": False,#True,
                     },
                     "framework": "torch",# if args.torch else "tf",
                     "num_workers": num_workers, # parallelism
                     "num_gpus":num_gpus,
                     "num_gpus_per_worker":num_gpus_per_worker,
                     "seed" : 0,
                     "target_noise_clip" : 0.5,#grid_search([-0.5,0.5]),
                     "learning_starts" : 500,#grid_search([500,1000,1500]),


                     "log_level": "WARN",
                    
                     "num_cpus_for_driver": 1,
                     "num_cpus_per_worker": 1,

                     "train_batch_size": 128,
                     "lr": grid_search(lr_lst),
                     "multiagent": {
                         "policies": policy_graphs,
                         "policy_mapping_fn": policy_mapping_fn,
                     },
                         }
        if algo == 'a2c' or algo == 'a3c':
            self.config={        

                   "env": "caching_vM",
                   "env_config": {
                   "ttl_var": ttl_var,
                   "variable": variable,
                   "num_agents": num_agents,
                   "nei_tab": ret_nei(cpt),
                   "lst_tab": ret_lst(cpt),
                   },

                   "model": {
                       "fcnet_hiddens": grid_search(fcnet_hidd_lst),#[64,64,64],
                       "fcnet_activation": grid_search(fcnet_act_lst),#"relu",
                       "vf_share_layers": False,#True,
                   },

                   "framework": "torch",# if args.torch else "tf",
                   "num_workers": 1,#num_workers, # parallelism
                   "num_gpus":num_gpus,
                   "num_gpus_per_worker":num_gpus_per_worker,
                   "seed" : 0,
                   "vf_loss_coeff" : 1,#grid_search([0.5,1.0]),
                   "grad_clip" : 30,#grid_search([30,40,50]),

                   "log_level": "WARN",
                  
                   "num_cpus_for_driver": 1,
                   "num_cpus_per_worker": 1,

                  
                   "train_batch_size": 128,
                   "lr": grid_search(lr_lst),
                   "multiagent": {
                       "policies": policy_graphs,
                       "policy_mapping_fn": policy_mapping_fn,
                   },
                       } 

        #if algo == "ppo" or algo == "appo":
        #  self.config["num_sgd_iter"]=10

       
        self.save_dir = "~/ray_results"
        self.stop_criteria = {
                    "training_iteration": stop_iters,#args.stop_iters,
                    "timesteps_total": stop_timesteps,#args.c,
                    "episode_reward_mean": stop_reward#args.stop_reward,
                    }

        def env_creator(_):
          return caching_vM(self.config)
        register_env(env_name, env_creator) #



    def train(self, algo):
        """
        Train an RLlib IMPALA agent using tune until any of the configured stopping criteria is met.
            See https://docs.ray.io/en/latest/tune/api_docs/execution.html#tune-run
        :return: Return the path to the saved agent (checkpoint) and tune's ExperimentAnalysis object
            See https://docs.ray.io/en/latest/tune/api_docs/analysis.html#experimentanalysis-tune-experimentanalysis
        """

        if algo == "ppo":
            analysis = ray.tune.run(ppo.PPOTrainer, name="my_exp", config=self.config,  local_dir=self.save_dir, stop=self.stop_criteria,
                               checkpoint_at_end=True )#, global_checkpoint_period=np.inf)
        if algo == "impala":
            analysis = ray.tune.run(impala.ImpalaTrainer, name="my_exp", config=self.config, local_dir=self.save_dir, stop=self.stop_criteria,
                                checkpoint_at_end=True)
        if algo == "a3c":
            analysis = ray.tune.run(a3c.A3CTrainer, name="my_exp", config=self.config, local_dir=self.save_dir, stop=self.stop_criteria,
                                checkpoint_at_end=True)
        if algo == "appo":
            analysis = ray.tune.run(ppo.APPOTrainer, name="my_exp", config=self.config, local_dir=self.save_dir, stop=self.stop_criteria,
                                checkpoint_at_end=True)
        if algo == "ddpg":
            analysis = ray.tune.run(ddpg.DDPGTrainer,name="my_exp", config=self.config, local_dir=self.save_dir, stop=self.stop_criteria,
                                checkpoint_at_end=True, keep_checkpoints_num= 1)#, global_checkpoint_period=np.inf)
        if algo == "td3":
            analysis = ray.tune.run(ddpg.TD3Trainer, name="my_exp", config=self.config, local_dir=self.save_dir, stop=self.stop_criteria,
                                checkpoint_at_end=True)
        
        lr = analysis.get_best_config(metric='episode_reward_mean', mode="max")["lr"] 
        fc_hid = analysis.get_best_config(metric='episode_reward_mean', mode="max")["model"]["fcnet_hiddens"] 
        fc_act = analysis.get_best_config(metric='episode_reward_mean', mode="max")["model"]["fcnet_activation"] 

        if algo=='ppo'  or algo == 'appo':
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
        checkpoints = analysis.get_trial_checkpoints_paths(trial=analysis.get_best_trial('episode_reward_mean', mode = 'max') ,
                                                           metric='episode_reward_mean')

        # retriev the checkpoint path; we only have a single checkpoint, so take the first one
        checkpoint_path = checkpoints[0][0]
        print("Checkpoint path:", checkpoint_path)
        print("lr = ", lr, " fc hid = ", fc_hid, " fc_act = ", fc_act)

        if algo =="ppo" or algo == 'appo':
            return checkpoint_path, analysis, lr, fc_hid, fc_act, vf_loss, kl_tar, clip
        if algo =="ddpg" or algo == "td3":
            return checkpoint_path, analysis, lr, fc_hid, fc_act, target_noise, lea_starts
        if algo =="a2c" or algo == "a3c":
            return checkpoint_path, analysis, lr, fc_hid, fc_act, vf_loss, grad


    def test(self,algo, path, lr, fc_hid, fc_act, vf_loss, kl_target, clip,  target_noise, lea_starts, grad):


        """Test trained agent for a single episode. Return the episode reward"""
        # instantiate env class
        unused_shared = []
        unused_own = []
        unsatisfied_shared = []
        unsatisfied_own = []

        episode_reward = 0
        
        #self.config["num_workers"] = 0
               
        #self.config["nei_tab"] = ret_nei(11)
        #self.config["lst_tab"] = ret_lst(11)
        self.config["lr"] = lr
        self.config['model']["fcnet_hiddens"] = fc_hid
        self.config['model']["fcnet_activation"] = fc_act

        if algo == 'ppo' or algo== 'appo':
            self.config["vf_loss_coeff"] = vf_loss
            self.config["kl_target"] = kl_target
            self.config["clip_param"] = clip
        if algo == 'ddpg' or algo == 'td3':
            self.config["target_noise_clip"] = target_noise
            self.config["learning_starts"] = lea_starts
        if algo == 'a2c' or algo == 'a3c':
            self.config["vf_loss_coeff"] = vf_loss
            self.config["grad_clip"] = grad       

        if algo == "ppo":
            self.agent = ppo.PPOTrainer(config=self.config)
        if algo == "ddpg":
            self.agent = ddpg.DDPGTrainer(config=self.config)
        if algo == "a3c":
            self.agent = a3c.A3CTrainer(config=self.config)
        if algo == "impala":
            self.agent = impala.ImpalaTrainer(config=self.config)
        if algo == "appo":
            self.agent = ppo.APPOTrainer(config=self.config)
        if algo == "td3":
            self.agent = ddpg.TD3Trainer(config=self.config)
        
        self.agent.restore(path)

        env = caching_vM(config=self.config)   
        
        obs = env.reset()
        done = False

        action = {}
        for agent_id, agent_obs in obs.items():
            policy_id = self.config['multiagent']['policy_mapping_fn'](agent_id)
            action[agent_id] = self.agent.compute_action(agent_obs, policy_id=policy_id)
        obs, reward, done, info = env.step(action)
        done = done['__all__']

        for x in range(20):#len(info)):
          #print(x,  "////////////////////////////////////////////////////////////////////")

          res = ast.literal_eval(info[x])
          unused_shared.append(res[0])
          unused_own.append(res[1])
          unsatisfied_shared.append(res[2])
          unsatisfied_own.append(res[3])

        #print("unused_own = ", unused_own , "////////////////////////////////////////////////////")
        #print("len unused_own = ", len(unused_own) , "////////////////////////////////////////////////////")

        print("reward == ", reward)
        # sum up reward for all agents
        episode_reward += sum(reward.values())

        return episode_reward, unused_shared, unused_own, unsatisfied_shared, unsatisfied_own



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default= 50)
    parser.add_argument("--stop-timesteps", type=int, default=90000000)
    parser.add_argument("--stop-reward", type=float, default=99999999999)
    parser.add_argument("--ttl_var", type=float, default=3)
    parser.add_argument("--cpt", type=int, default=1)
    parser.add_argument("--run", type=str, default="ppo") 
    parser.add_argument("--num_gpus_per_worker", type=float, default= 0)
    parser.add_argument("--num_workers", type=int, default= 0)
    parser.add_argument("--activation", nargs="+", default= ["relu"])
    parser.add_argument('-l','--layer', type=int, nargs='+', required=True, action='append', help='layer list')
    parser.add_argument("--lr", type=float, nargs="+", default=[1e-2])
    parser.add_argument("--gpu", type=float, default= 0)
    parser.add_argument("--cpu", type=int, default= 8)

    ray.shutdown()
    ray.init()#num_gpus=0)
    
    num_agents = 22

    args = parser.parse_args()

    #init  8,8,4   10,10,2
    exper = customExperimentClass(num_agents, args.run , args.ttl_var, args.cpt, [8,8,4], \
            fcnet_hidd_lst = args.layer, fcnet_act_lst = args.activation, lr_lst = args.lr, stop_iters=args.epochs, num_gpus=args.gpu, num_gpus_per_worker=args.num_gpus_per_worker, num_workers=args.num_workers)                                  
    
    #train model
    #checkpoint_path, results, lr, fc_hid, fc_act = exper.train(args.run)
    all_in = exper.train(args.run)
    checkpoint_path=all_in[0]
    results= all_in[1]
    lr= all_in[2]
    fc_hid= all_in[3]
    fc_act= all_in[4]

    if args.run== 'ppo' or args.run == 'appo':
        vf_loss=all_in[5]
        kl_target= all_in[6]
        clip=all_in[7]

    if args.run== 'ddpg' or args.run== 'td3':
        target_noise=all_in[5]
        lea_starts=all_in[6]

    if args.run== 'a2c' or args.run== 'a3c':
        vf_loss=all_in[5]
        grad=all_in[6]

    #plot reward (to test model learning and convergence)
    #the_plot(results)

    #test model
    #reward , unused_shared ,unused_own, unsatisfied_shared, unsatisfied_own = exper.test(args.run ,checkpoint_path, lr, fc_hid, fc_act) 
    if args.run== 'ppo' or args.run== 'appo':
        reward, unused_shared ,unused_own, unsatisfied_shared, unsatisfied_own  = exper.test(args.run ,checkpoint_path, lr, fc_hid, fc_act, vf_loss, kl_target, clip, -1, -1,-1)
    if args.run== 'ddpg' or args.run== 'td3':
        reward, unused_shared ,unused_own, unsatisfied_shared, unsatisfied_own  = exper.test(args.run ,checkpoint_path, lr, fc_hid, fc_act, -1, -1, -1, target_noise, lea_starts, -1)
    if args.run== 'a2c' or args.run== 'a3c':
        reward, unused_shared ,unused_own, unsatisfied_shared, unsatisfied_own  = exper.test(args.run ,checkpoint_path, lr, fc_hid, fc_act, vf_loss,    -1,      -1,   -1, -1,  grad)


    #print the best parameters
    print(" best lr = ", lr)
    print(" best hidden layer parameter = ", fc_hid)
    print(" best fonction activation  = ", fc_act)

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
    