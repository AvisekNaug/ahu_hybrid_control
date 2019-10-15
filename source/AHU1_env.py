# importing the modules for reading the data and pre-processing it
from pandas import *
import numpy as np
from joblib import dump, load
from helperfunctions import merge_df_rows

# importing the modules for setting up the environment from which the
# algorithm learns the control policy to be implemented
import gym
from gym import spaces
from gym.utils import seeding

# Importing my own packages which may contain some bugs
import Energy_Calc

# This class describes the formal environment which the reinforcement learning
# interacts with. It inherits some properties from the gym imported earlier
class Env(gym.Env):

    def __inint__(self, datapath, modelpath: list):

        # data to be used for training and testing
        self.traindataset = read_pickle(datapath)

        # State Space variables
        self.statespace = ['OAT', 'RH', 'AirFlow', 'PH_T', 'SAT']

        # shape of dataframe
        self.m, self.n = self.traindataset.shape

        # slicing data into train and test sequences
        self.slicepoint = 0.75
        self.traindatalimit = int(self.slicepoint*self.m)
        self.testdatalimit = self.m

        # getting 0:mean 1:std 2:min 3:max- maintain a dataframe as much
        # as possible
        self.Stats = self.traindataset[self.statespace].describe().iloc[[1, 2, 3, 7], :-1]

        # Windowed Stats: Assuming a window of 24 hours
        self.win_len = int(1440 / (5 * period))
        self.windowMean = self.traindataset.rolling(self.win_len, min_periods=1).mean()['OAT']
        self.windowMax = self.traindataset.rolling(self.win_len, min_periods=1).max()['OAT']
        self.windowMin = self.traindataset.rolling(self.win_len, min_periods=1).min()['OAT']

        # Standard requirements for interfacing with Keras-RL's code
        SpaceLB = [self.Stats[2, i] for i in range(self.n - 1)]
        SpaceUB = [self.Stats[3, i] for i in range(self.n - 1)]
        self.observation_space = spaces.Box(low=np.array(SpaceLB),
                                            high=np.array(SpaceUB),
                                            dtype=np.float32)
        # self.action_space = spaces.Box(low=np.array([self.Stats[2,3]]), high=np.array([self.Stats[3,3]]),
        #                              dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([55.0]), high=np.array([75.0]),
                                       dtype=np.float32)
        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

