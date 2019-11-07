# import modules
import numpy as np
import tensorflow as tf
from stable_baselines import TD3
from stable_baselines.td3.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.bench import Monitor

"""
Defines functions that construct various components of a reinforcement learning
agent
"""


def get_agent(env, rllogs):
    """
    Generate a `Twin Delayed DDPGAgent` instance that TD3 is a direct successor of DDPG
    and improves it using three major tricks: clipped double Q-Learning, delayed policy
    update and target policy smoothing
    """

    # Wrap the environment for monitoring
    env = Monitor(env, rllogs)

    # TD3 takes the environment in vectorized form
    env = DummyVecEnv([lambda: env])

    # necessary steps to adjust the env for testing
    env.env_method("testenv")
    # env.set_attr("testing", True)
    env.set_attr("dataPtr", 10000)

    # The noise objects for TD3
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # create a TD3 agent with the above parameters
    agent = TD3(MlpPolicy, env, action_noise=action_noise, verbose=1, tensorboard_log="../td3_hvac_tensorboard/")

    return agent


def train_agent(agent, rllogs, env=None, steps=30000):
    """
    Use a `DDPGAgent` instance to train its policy. The agent stores the best
    policy it encounters during training for use later. Once trained, thw agent
    can be used to exploit its experience.
    """

    # used in case of incremental learning
    if env is not None:
        # Wrap the environment for monitoring
        env = Monitor(env, rllogs)

        # TD3 takes the environment in vectorized form
        env = DummyVecEnv([lambda: env])

        # set this as the environment
        agent.set_env(env)

    agent.learn(total_timesteps=steps, callback=SaveBest, tb_log_name="results_run")


def test_agent(agent, env, rllogs_local, episodes = 1):
    """
    Run the agent in an environment and store the actions it takes in a list.
    """

    # Wrap the environment for monitoring
    env = Monitor(env, rllogs_local)

    # TD3 takes the environment in vectorized form
    env = DummyVecEnv([lambda: env])

    # necessary steps to adjust the env for testing
    env.env_method("testenv")
    # env.set_attr("testing", True)
    env.set_attr("dataPtr", 10000)

    perf_metrics = performancemetrics()

    for _ in range(episodes):
        perf_metrics.on_episode_begin()
        obs = env.reset()
        dones = False
        while not dones:
            action, _ = agent.predict(obs)
            obs, rewards, dones, info = env.step(action)
            perf_metrics.on_step_end(info[0])
        perf_metrics.on_episode_end()

    return perf_metrics

best_mean_reward = -np.inf
n_steps = 0
rllogs = '../RL_data/'

def SaveBest(_locals, _globals):
    """
    Store neural network weights during training if the current episode's
    performance is better than the previous best performance.

    Args:
    * `dest`: name of `h5f` file where to store weights.
    """

    global best_mean_reward, n_steps, rllogs

    # Print stats every 1000 calls
    if (n_steps + 1) % 1000 == 0:
        # Evaluate policy training performance
        x, y = ts2xy(load_results(rllogs), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            print(x[-1], 'timesteps')
            print(
                "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                _locals['self'].save(rllogs + 'best_model.pkl')
    n_steps += 1

    # logging the variable weights histogram
    # Create summaries to visualize weights
    self_ = _locals['self']
    for var in self_.policy_out.policy:
        tf.summary.histogram(var.name, var)


    return True


class performancemetrics():
    """
    Store the history of performance metrics. Useful for evaluating the
    agent's performance:
    """

    def __init__(self, metrics=[]):
        self.metrics = metrics  # store perf metrics for each episode
        self.metric = {}

    def on_episode_begin(self, logs={}):
        self.metric = logs  # store performance metrics

    def on_episode_end(self):
        self.metrics.append(self.metric)

    def on_step_end(self, info={}):
        for key, value in info.items():
            if key in self.metric:
                self.metric[key].append(value)
            else:
                self.metric[key] = [value]
