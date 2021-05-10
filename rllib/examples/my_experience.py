"""Example of a custom experiment wrapped around an RLlib trainer."""
import argparse

import ray
from ray import tune
from ray.rllib.agents import ppo
from my_env import ContentCaching
import pickle
import time
import numpy as np


unused_shared_step =[-1]
unused_own_step =[-1]
unsatisfied_shared_step =[-1]
unsatisfied_own_step =[-1]

parser = argparse.ArgumentParser()
#parser.add_argument("--train-iterations", type=int, default=10)
#parser.add_argument("--ttl_var", type=int, default=3)
#parser.add_argument("--stop-iters", type=int, default= 20)

#unused_shared_step =[]
#unused_own_step =[]
#unsatisfied_shared_step =[]
#unsatisfied_own_step =[]


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




def experiment(config):
    
    #global unused_shared_step
    #global unused_own_step
    #global unsatisfied_shared_step
    #global unsatisfied_own_step

    

    iterations = 2
    train_agent = ppo.PPOTrainer(config=config, env=ContentCaching)#"ContentCaching-v0")
    checkpoint = None
    train_results = {}

    # Train
    #iterations = 20
    for i in range(iterations):
        train_results = train_agent.train()
        if i % 2 == 0 or i == iterations - 1:
            checkpoint = train_agent.save(tune.get_trial_dir())
        tune.report(**train_results)
    train_agent.stop()

    # Manual Eval
    config["num_workers"] = 0
    eval_agent = ppo.PPOTrainer(config=config, env=ContentCaching)#"ContentCaching-v0")
    eval_agent.restore(checkpoint)
    env = eval_agent.workers.local_worker().env

    obs = env.reset()
    done = False
    eval_results = {"eval_reward": 0, "eval_eps_length": 0}
    while not done:
        action = eval_agent.compute_action(obs)
        next_obs, reward, done, info,  = env.step(action)

        #unused_shared_step.append(info["unused_shared"])
        #unused_own_step.append(info["unused_own"])
        #unsatisfied_shared_step.append(info["unsatisfied_shared"])
        global unsatisfied_own_step
        unsatisfied_own_step= 99#.append(10)#info["unused_own"])


        print(" info[unused_shared] =xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx ", info["unused_shared"] )

        eval_results["eval_reward"] += reward
        eval_results["eval_eps_length"] += 1
    results = {**train_results, **eval_results}
    tune.report(results)

    


if __name__ == "__main__":

    args = parser.parse_args()
    
    ray.init(num_cpus=3)#num_cpus=3
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

    results = tune.run(
        experiment,
        config=config,
        resources_per_trial=ppo.PPOTrainer.default_resource_request(config) )#, local_dir="~/ray_results", resume=True)
    
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx---------------------------------------------------------------------------")

    #analysis = tune.run(my_trainable, local_dir="~/tune_results", resume=True)

    #print("unused_shared_step = " ,unused_shared_step)
    #print("unused_own_step = " ,np.mean(unused_own_step))
    #print("unsatisfied_shared_step  = " ,np.mean(unsatisfied_shared_step))
    print("unsatisfied_own_step = " ,np.mean(unsatisfied_own_step))

    #print("results ", results)
    #analysis = ExperimentAnalysis( experiment_checkpoint_path="~/ray_results/my_exp/state.json")
    
