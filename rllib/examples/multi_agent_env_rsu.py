import gym
import numpy as np

from ray.rllib.env.multi_agent_env import MultiAgentEnv


import ray
from ray import tune
from ray.rllib.agents.registry import get_agent_class
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer


import random



cpt = 1



class caching_vM(MultiAgentEnv):
    def __init__(self, config, return_agent_actions = False, part=False):
        self.num_agents = 22
        self.observation_space = gym.spaces.Box(low=0, high=200, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)


        #print("config ==== ", config)
        self.ttl_var = config["env_config"]["ttl_var"]
        self.variable = config["env_config"]["variable"]

        self.neighbor = config["env_config"]["nei_tab"]
        self.request = config["env_config"]["lst_tab"]

        self.reward_cumul = []

        lst = self.request

        tab_cache= []
        tab_request = []
        nei_req = []
        cache_on_tab = []
        neighbor_number_tab = []
        ttl_tab = []

        #vehicles
        for xx in range(20):
            tab_cache.append(50) 
            tab_request.append(lst[xx])
            nei_req.append(-99)
            cache_on_tab.append(0)
            neighbor_number_tab.append(0)
            ttl_tab.append(np.zeros(20))
        #RSU
        for xx in range(20,22):
            tab_cache.append(200) 
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
        self.steps = 0

    def next_obs(self,i):
        
        #if i == 3:
            #self.steps = self.steps+1
            #print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx == ", self.steps)
        nei_tab = self.neighbor#ret_nei(self.cpt)# args must be variable
        ttl_var = self.ttl_var # must be variable
        
        entity_pos = {}

        for x in range(len(self.caching_cap)):
            lstt= []

            lstt.append(self.request[x][i])

            #init  caching_cap
            if i == 0 :
                if x==20 or x== 21:
                    
                    self.caching_cap[x]=200
                    lstt.append(200)
                else:
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

                if len(nei_tab[i][y]) == 0 or len(nei_tab[i][nei_tab[i][x][y]]) == 0:
                    cache = cache + 0
                
                else:
                    cache = cache + (self.request[nei_tab[i][x][y]][i]/len(nei_tab[i][nei_tab[i][x][y]]) )

            if len(nei_tab[i][x])==0:
                self.neigbors_request[x]= 0
                self.neighbor_number[x] = 0
                lstt.append(0)
            else:
                self.neigbors_request[x] = cache/len(nei_tab[i][x])
                self.neighbor_number[x] = len(nei_tab[i][x])
                lstt.append(self.neigbors_request[x])  

            entity_pos[x] = np.array(lstt)

        #entity_pos = np.array(entity_pos)
        return entity_pos

    def reset(self):
        self.epochs_num=0
        self.dones = set()
        entity_pos = self.next_obs(0)
        return entity_pos


    def step(self, action):

        obs, rew, done, info = {}, {}, {}, {}

        nei_tab = self.neighbor#ret_nei(self.cpt)
        self.epochs_num= self.epochs_num+1
        i = self.epochs_num
        entity_pos = self.next_obs(self.epochs_num)
        variable = self.variable 
        reward=[]
        R_c = variable[0]
        C = variable[1]
        fact_k = variable[2] 

        unused_shared = []
        unused_own = []
        nei_request_tab = []
        unsatisfied_shared = []
        unsatisfied_own =[]

        for zz in range(self.num_agents):
            cache1 = 0
            for y in range(len(nei_tab[i][zz])):

                if len(nei_tab[i][y]) == 0 or len(nei_tab[i][nei_tab[i][zz][y]]) == 0:
                    cache1= cache1 + 0
                else :
                    cache1=cache1+(max(0,(self.request[nei_tab[i][zz][y]][i]-((1-action[nei_tab[i][zz][y]][0])*self.caching_cap[nei_tab[i][zz][y]]))/len(nei_tab[i][nei_tab[i][zz][y]])) )
   
            if len(nei_tab[i][zz]) == 0 :
                cache1 = 0
           
            
            f = R_c * max(0, (1-action[zz][0]) * self.caching_cap[zz] )  \
               - C* ( max(0,  (self.request[zz][i]-(action[zz][0]*self.caching_cap[zz]))) + max(0, ( cache1 - (1-action[zz][0])*self.caching_cap[zz])/fact_k)  ) \
                  - C* ( max(0, ((action[zz][0]*self.caching_cap[zz])-self.request[zz][i])/fact_k) + max (0, ((1-action[zz][0])*self.caching_cap[zz]) - cache1) )  
            
            #unused_shared.append( float(max(0,(1-action[zz][0])*self.caching_cap[zz] - cache1  )))
            #unused_own.append( float(max(0, (action[zz][0] * self.caching_cap[zz])-self.request[zz][i] )))
            #unsatisfied_shared.append(float(max(0,cache1 - (1-action[zz][0])*self.caching_cap[zz])))
            #unsatisfied_own.append(float(max(0,self.request[zz][i] - action[zz][0]*self.caching_cap[zz])))

            rew[zz] = f

            info[zz] = str( [float(max(0,(1-action[zz][0])*self.caching_cap[zz] - cache1 )),\
                        float(max(0, (action[zz][0] * self.caching_cap[zz])-self.request[zz][i] )), \
                            float(max(0,cache1 - (1-action[zz][0])*self.caching_cap[zz])), \
                                float(max(0,self.request[zz][i] - action[zz][0]*self.caching_cap[zz]))]) 

            #print(" info[zz] = = ", info[zz])

            done[zz] = True

        #init  self.cache_on[x]
        for zz in range(len(action)):
            self.cache_on[zz] = min(self.request[zz][i], ((action[zz][0]*100) * self.caching_cap[zz]) / 100.0)  \
                + min(self.neigbors_request[zz], (((1-action[zz][0])*100) * self.caching_cap[zz]) / 100.0)
        


        done["__all__"] = True


        """thisdict = {
                              "unused_shared": np.mean(unused_shared),
                              "unused_own": np.mean(unused_own),
                              "unsatisfied_shared": np.mean(unsatisfied_shared),
                              "unsatisfied_own": np.mean(unsatisfied_own)
                            }"""

        done["__all__"] = True
        
        return entity_pos, rew, done, info#thisdict



