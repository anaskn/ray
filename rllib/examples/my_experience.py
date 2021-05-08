"""Example of a custom experiment wrapped around an RLlib trainer."""
import argparse

import ray
from ray import tune
from ray.rllib.agents import ppo
from my_env import ContentCaching
import pickle
import time

parser = argparse.ArgumentParser()
#parser.add_argument("--train-iterations", type=int, default=10)
#parser.add_argument("--ttl_var", type=int, default=3)
#parser.add_argument("--stop-iters", type=int, default= 20)


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
    iterations = 10#config.pop("train-iterations")
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
    config["num_workers"] = 1
    eval_agent = ppo.PPOTrainer(config=config, env=ContentCaching)#"ContentCaching-v0")
    eval_agent.restore(checkpoint)
    env = eval_agent.workers.local_worker().env

    obs = env.reset()
    done = False
    eval_results = {"eval_reward": 0, "eval_eps_length": 0}
    while not done:
        action = eval_agent.compute_action(obs)
        next_obs, reward, done, info,  = env.step(action)
        eval_results["eval_reward"] += reward
        eval_results["eval_eps_length"] += 1
    #results = {**train_results, **eval_results}
    #tune.report(results)
    


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

    tune.run(
        experiment,
        config=config,
        resources_per_trial=ppo.PPOTrainer.default_resource_request(config))

    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx---------------------------------------------------------------------------")
