"""
Defines functions that construct various components of a reinforcement learning
agent
"""
# from typing import List, Any

from keras import backend as K
from keras.models import Sequential, Model
from keras.callbacks import Callback
from keras.layers import Dense, Activation, Flatten, Input, Concatenate, Multiply, Lambda, BatchNormalization
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess


def get_agent(env) -> DDPGAgent:
    """
    Generate a `DDPGAgent` instance that represents an agent learned using
    Deep Deterministic Policy Gradient. The agent has 2 neural networks: an actor
    network and a critic network.

    Args:
    * `env`: An OpenAI `gym.Env` instance

    Returns:
    * a `DDPGAgent` instance.
    """
    assert len(env.action_space.shape) == 1
    nb_actions = env.action_space.shape[0]
    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')

    range_action_input = 0.5 * (env.action_space.high - env.action_space.low)
    constantBias = 1
    lowb = env.action_space.low

    # actor = Flatten(input_shape=(1,) + env.observation_space.shape)(observation_input)
    y = Flatten()(observation_input)
    y = Dense(16)(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Dense(16)(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    pht = Dense(1)(y)
    pht = BatchNormalization()(pht)
    pht = Activation('tanh')(pht)
    pht = Lambda(lambda a: (a + K.constant(constantBias)) * K.constant(range_action_input[0])
                           + K.constant(lowb[0]))(pht)
    rht = Dense(1)(y)
    rht = BatchNormalization()(rht)
    rht = Activation('tanh')(rht)
    rht = Lambda(lambda a: (a + K.constant(constantBias)) * K.constant(range_action_input[1])
                           + K.constant(lowb[1]))(rht)
    axn = Concatenate()([pht, rht])
    actor = Model(inputs=observation_input, outputs=axn)

    flattened_observation = Flatten()(observation_input)
    x = Concatenate()([action_input, flattened_observation])
    x = Dense(32)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)

    memory = SequentialMemory(limit=1000, window_length=1)

    random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.5, size=nb_actions)
    agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                      memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                      gamma=.99, target_model_update=1e-3, random_process=random_process)
    agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])
    return agent


class SaveBest(Callback):
    """
    Store neural network weights during training if the current episode's
    performance is better than the previous best performance.

    Args:
    * `dest`: name of `h5f` file where to store weights.
    """

    def __init__(self, dest: str):
        super().__init__()
        self.dest = dest
        self.lastreward = -1000000
        self.rewardsTrace = []

    def on_episode_end(self, episode, logs={}):
        self.rewardsTrace.append(logs.get('episode_reward'))
        if logs.get('episode_reward') > self.lastreward:
            self.lastreward = logs.get('episode_reward')
            self.model.save_weights(self.dest, overwrite=True)


def train_agent(agent, env, steps=30000, dest='agent_weights.h5f'):
    """
    Use a `DDPGAgent` instance to train its policy. The agent stores the best
    policy it encounters during training for use later. Once trained, the agent
    can be used to exploit its experience.

    Args:
    * `agent`: A `DDPGAgent` returned by `get_agent()`.
    * `env`: An OpenAI `gym.Env` environment in which the agent will operate.
    * `steps`: Number of actions to train over. The larger the number, the more
    experience the agent uses to learn, and the longer training takes.
    * `dest`: name of `h5f` file where to store weights.

    Retruns
    * `store_weights`: Containing th reward trace
    """
    train_metrics = SaveBest(dest=dest)
    agent.fit(env, nb_steps=steps, visualize=False, verbose=0, callbacks=[train_metrics])
    return train_metrics


class PerformanceMetrics(Callback):
    """
    Store the history of performance metrics. Useful for evaluating the
    agent's performance:
    """

    def __init__(self, metrics=[]):
        self.metrics = [] # store perf metrics for each episode
        super().__init__()

    def on_episode_begin(self, episode, logs={}):
        self.metric = {}  # store performance metrics

    def on_episode_end(self, episode, logs={}):
        self.metrics.append(self.metric)

    def on_step_end(self, step, logs={}):
        for key, value in logs.get('info').items():
            if key in self.metric:
                self.metric[key].append(value)
            else:
                self.metric[key] = [value]


def test_agent(agent, env, weights='agent_weights.h5f', actions=[]) -> \
        PerformanceMetrics:
    """
    Run the agent in an environment and store the actions it takes in a list.

    Args:
    * `agent`: A `DDPGAgent` returned by `get_agent()`.
    * `env`: An OpenAI `gym.Env` environment in which the agent will operate.
    * `actions`: A list in which to store actions.

    Returns:
    * The list containing history of actions.
    """
    test_perf_log = PerformanceMetrics(actions)
    agent.load_weights(weights)
    agent.test(env, nb_episodes=1, visualize=False, verbose=0, callbacks=[test_perf_log])
    return test_perf_log
