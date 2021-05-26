from ray.rllib.agents import impala
import gym
import ray
from ray import tune

class customExperimentClass():
    def __init__(self):
        ray.shutdown()
        ray.init(num_cpus=2, num_gpus=0)
        self.env = gym.make("CartPole-v0")
        
        self.config = dict(**{
                        "env": "CartPole-v0",
                        #"callbacks": MyCallbacks,
                        "num_workers": 0,
                        "num_gpus": 0,
                        "evaluation_num_workers": 1,
                        # Custom eval function
                        #"custom_eval_function": custom_eval_function,
                        # Enable evaluation, once per 100 training iteration.
                        "evaluation_interval": 1,
                        # Run 3 episodes each time evaluation runs.
                        "evaluation_num_episodes": 1,
                        "model": {
                            # Nonlinearity for fully connected net (tanh, relu)
                            "fcnet_activation": "tanh",
                            # Number of hidden layers for fully connected net
                            "fcnet_hiddens": [16,8],
                            # Whether to wrap the model with an LSTM.
                            # Needs LSTMWrapper.get_initial_state() ?
                            #"use_lstm": True,           # <--------------------- Change this to see it working.
                            # Max seq len for training the LSTM, defaults to 20.
                            #"max_seq_len": 20,
                            # Size of the LSTM cell.
                            #"lstm_cell_size": 16,
                            # Whether to feed a_{t-1}, r_{t-1} to LSTM.
                            #"lstm_use_prev_action_reward": False, 
                            },
                            "framework": "tf",
                            "ignore_worker_failures": True,
                    })
    
    def train(self):
        """
        Train an RLlib IMPALA agent using tune until any of the configured stopping criteria is met.
            See https://docs.ray.io/en/latest/tune/api_docs/execution.html#tune-run
        :return: Return the path to the saved agent (checkpoint) and tune's ExperimentAnalysis object
            See https://docs.ray.io/en/latest/tune/api_docs/analysis.html#experimentanalysis-tune-experimentanalysis
        """
        #analysis = ray.tune.run(ppo.PPOTrainer, config=self.config, local_dir=self.save_dir, stop=stop_criteria,
        #                        checkpoint_at_end=True)
        # list of lists: one list per checkpoint; each checkpoint list contains 1st the path, 2nd the metric value
        results = tune.run("IMPALA",
                    verbose=1,
                    config=self.config,
                    stop={"training_iteration":  1,},
                    checkpoint_freq=1,
                    keep_checkpoints_num=1,
                    checkpoint_score_attr='training_iteration',
                   )
        checkpoints = results.get_trial_checkpoints_paths(trial=results.get_best_trial('episode_reward_mean',mode="max"),
                                                           metric='episode_reward_mean')
        # retriev the checkpoint path; we only have a single checkpoint, so take the first one
        checkpoint_path = checkpoints[0][0]
        print("Checkpoint path:", checkpoint_path)
        return checkpoint_path, results

    def load(self, path):
        """
        Load a trained RLlib agent from the specified path. Call this before testing a trained agent.
        :param path: Path pointing to the agent's saved checkpoint (only used for RLlib agents)
        """
        self.agent = impala.ImpalaTrainer(config=self.config)
        self.agent.restore(path)

    def test(self):
        """Test trained agent for a single episode. Return the episode reward"""
        # instantiate env class
        #env = self.env

        # run until episode ends
        episode_reward = 0
        done = False
        obs = self.env.reset()
        while not done:
            action = self.agent.compute_action(obs)
            obs, reward, done, info = self.env.step(action)
            episode_reward += reward

        return episode_reward


# Class instance
exper = customExperimentClass()
# Train and save for 2 iterations
checkpoint_path, results = exper.train()
# Load saved
exper.load(checkpoint_path)
# Test loaded
exper.test()