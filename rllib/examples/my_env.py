"""Example of a custom gym environment and model. Run this for a demo.

This example shows:
  - using a custom environment
  - using a custom model
  - using Tune for grid search

You can visualize experiment results in ~/ray_results using TensorBoard.
"""
import argparse
import gym
from gym.spaces import Discrete, Box
import numpy as np
import os
import random

import ray
from ray import tune
from ray.tune import grid_search
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
import pickle

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")
parser.add_argument("--torch", action="store_true")
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--stop-iters", type=int, default= 20)#50)
parser.add_argument("--stop-timesteps", type=int, default=100000)
parser.add_argument("--stop-reward", type=float, default=0.1)

parser.add_argument("--ttl_var", type=int, default=3)




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


class ContentCaching(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.

    You can configure the length of the corridor via the env config."""

   

    def __init__(self, config: EnvContext):
        self.action_space = Box(low= 0, high= 1 ,shape=(20,), dtype=np.float32)
        self.observation_space = Box(low= 0, high= 100, shape=(20,3), dtype=np.float32)
        # Set the seed. This is only used for the final (reach goal) reward.
        #self.seed(config.worker_index * config.num_workers)

        self.ttl_var = config["ttl_var"]
        self.variable = config["variable"]
        self.cpt = config["cpt"]

        self.neighbor = config["nei_tab"]
        self.request = config["lst_tab"]

        lst = self.request#ret_lst(self.cpt)

        tab_cache= []
        tab_request = []
        nei_req = []
        cache_on_tab = []
        neighbor_number_tab = []
        ttl_tab = []
        for xx in range(20):
            tab_cache.append(50) 
            tab_request.append(lst[xx])
            nei_req.append(-99)
            cache_on_tab.append(0)
            neighbor_number_tab.append(0)
            ttl_tab.append(np.zeros(20))
        
        self.caching_cap =  tab_cache 
        self.request = tab_request
        self.neigbors_request = nei_req
        self.cache_on = cache_on_tab
        self.neighbor_number = neighbor_number_tab
        self.ttl = ttl_tab

        self.unused_shared=None 
        self.unused_own = None
        self.unsatisfied_shared= None
        self.unsatisfied_own= None

        self.epochs_num=0

    def next_obs(self,i):
        nei_tab = self.neighbor#ret_nei(self.cpt)# args must be variable
        ttl_var = self.ttl_var # must be variable
        
        entity_pos = []

        for x in range(len(self.caching_cap)):
            lstt= []
            lstt.append(self.request[i][x])

            #init  caching_cap
            if i == 0 :
                self.caching_cap[x]=50
                lstt.append(50)
            else:           
                if i-ttl_var > 0:
                    self.caching_cap[x] = self.caching_cap[x] + self.ttl[x][i-ttl_var]
                    
                min_val = min(self.caching_cap[x] , float(self.cache_on[x]))
                self.caching_cap[x] = self.caching_cap[x] - min_val  

                self.ttl[x][i] = min_val
                lstt.append(self.caching_cap[x])
            
            #init  neigbors_request
            cache = 0
            for y in range(len(nei_tab[i][x])):

                if len(nei_tab[i][y]) == 0:
                    cache = cache + 0
                
                else:
                    cache = cache + (self.request[i][nei_tab[i][x][y]]/len(nei_tab[i][nei_tab[i][x][y]]) )

            if len(nei_tab[i][x])==0:
                self.neigbors_request[x]= 0
                self.neighbor_number[x] = 0
                lstt.append(0)
            else:
                self.neigbors_request[x] = cache/len(nei_tab[i][x])
                self.neighbor_number[x] = len(nei_tab[i][x])
                lstt.append(self.neigbors_request[x])  #cache/len(nei_tab[i][x])

            entity_pos.append(lstt)

        entity_pos = np.array(entity_pos)
        return entity_pos

    def reset(self):
        self.epochs_num=0
        entity_pos = self.next_obs(0)
        return entity_pos

    def step(self, action):

        nei_tab = self.neighbor#ret_nei(self.cpt)
        self.epochs_num= self.epochs_num+1
        i = self.epochs_num
        entity_pos = self.next_obs(self.epochs_num)
        variable = self.variable 
        reward=[]
        R_c = variable[0]
        C_o = variable[1]
        C_u = variable[2]
        fact_k = variable[3] 

        unused_shared = []
        unused_own = []
        nei_request_tab = []
        unsatisfied_shared = []
        unsatisfied_own =[]

        for zz in range(len(action)):
            cache1 = 0
            for y in range(len(nei_tab[i][zz])):

                if len(nei_tab[i][y]) == 0:
                    cache1= cache1 + 0
                else :
                    cache1=cache1+(max(0,(self.request[i][nei_tab[i][zz][y]]-((1-action[nei_tab[i][zz][y]])*self.caching_cap[nei_tab[i][zz][y]]))/len(nei_tab[i][nei_tab[i][zz][y]])) )
   
            if len(nei_tab[i][zz]) == 0 :
                cache1 = 0
           
            
            f = R_c * max(0, (1-action[zz]) * self.caching_cap[zz] )  \
               - C_u * ( max(0,  (self.request[i][zz]-(action[zz]*self.caching_cap[zz]))) + max(0, ( cache1 - (1-action[zz])*self.caching_cap[zz])/fact_k)  ) \
                  - C_o * ( max(0, ((action[zz]*self.caching_cap[zz])-self.request[i][zz])/fact_k) + max (0, ((1-action[zz])*self.caching_cap[zz]) - cache1) )  
        
            unused_shared.append( float(max(0,(1-action[zz])*self.caching_cap[zz] - cache1  )))
            unused_own.append( float(max(0, (action[zz]*self.caching_cap[zz])-self.request[i][zz] )))
            unsatisfied_shared.append(float(max(0,cache1-  (1-action[zz])*self.caching_cap[zz])))
            unsatisfied_own.append(float(max(0,self.request[i][zz] - action[zz]*self.caching_cap[zz])))

            reward.append(f)
        #init  self.cache_on[x]
        for zz in range(len(action)):
            self.cache_on[zz] = min(self.request[i][zz], ((action[zz]*100) * self.caching_cap[zz]) / 100.0)  \
                + min(self.neigbors_request[zz], (((1-action[zz])*100) * self.caching_cap[zz]) / 100.0)
        
        if self.epochs_num==19:
            done = True
        else:
            done = False

        thisdict = {
              "unused_shared": np.mean(unused_shared),
              "unused_own": np.mean(unused_own),
              "unsatisfied_shared": np.mean(unsatisfied_shared),
              "unsatisfied_own": np.mean(unsatisfied_own)
            }
        
        return entity_pos,np.mean(reward), done, thisdict

    def seed(self, seed=None):
        random.seed(seed)


class CustomModel(TFModelV2):
    """Example of a keras custom model that just delegates to an fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(CustomModel, self).__init__(obs_space, action_space, num_outputs,
                                          model_config, name)
        self.model = FullyConnectedNetwork(obs_space, action_space,
                                           num_outputs, model_config, name)

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()


class TorchCustomModel(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.torch_sub_model = TorchFC(obs_space, action_space, num_outputs,
                                       model_config, name)

    def forward(self, input_dict, state, seq_lens):
        input_dict["obs"] = input_dict["obs"].float()
        fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
        return fc_out, []

    def value_function(self):
        return torch.reshape(self.torch_sub_model.value_function(), [-1])


if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()

    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor(config))
    ModelCatalog.register_custom_model(
        "my_model", TorchCustomModel if args.torch else CustomModel)

    config = {
        "env": ContentCaching,  # or "corridor" if registered above
        "env_config": {
            "ttl_var": args.ttl_var,
            "cpt": 1,
            "variable": [8,8,8,4],
            "nei_tab": ret_nei(1),
            "lst_tab": ret_lst(1),

        },
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "model": {
            "custom_model": "my_model",
            "vf_share_layers": True,
        },
        "lr": grid_search([1e-2, 1e-4, 1e-6]),  # try different lrs
        "num_workers": 2,  # parallelism
        "framework": "torch" if args.torch else "tf",
    }

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    results = tune.run(args.run, config=config, stop=stop)
    

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)

    print("results ====== ", results)
    ray.shutdown()
