"""import argparse
import pickle 
import numpy as np

parser = argparse.ArgumentParser(description='...')
#parser.add_argument('-l','--layer', type=int, nargs='+', required=True, action='append', help='layer list')
parser.add_argument("--activation", nargs="+", default= ["relu"])

args = parser.parse_args()

#print(args.layer)
print(args.activation)
cpt = 1

string1 =  'data4/listfile_40_'+str(cpt)+'.data' #_evol'+ , _pos'+
with open(string1, 'rb') as filehandle:
# read the data as binary data stream
    lst = pickle.load(filehandle)


string2 = 'data4/nei_tab_pos_40_'+str(cpt)+'.data'
with open(string2, 'rb') as filehandle:
    # read the data as binary data stream
    nei_tab = pickle.load(filehandle)

print(np.shape(nei_tab)) # good (40,20)
print(np.shape(lst)) # good (40,20)


print(lst[19][19])"""

#!/usr/bin/env python
# encoding: utf-8

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
#from my_env import ContentCaching
import pickle
import time
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt 
from gym_example.envs.caching_env import Caching_v0




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




if __name__ == "__main__":

    lst = ret_lst(1)
    nei_tab = ret_nei(1)

    print("request vehicle 10 in step 5 == " , lst[10][5])
    print("")
    print("nei vehicle 10 in step 5 == " , nei_tab[10][5])