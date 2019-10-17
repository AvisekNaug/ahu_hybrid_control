# importing the modules for reading the data and pre-processing it
from pandas import *
import numpy as np
from joblib import dump, load
from . helperfunctions import merge_df_rows

# importing the modules for setting up the environment from which the
# algorithm learns the control policy to be implemented
import gym
from gym import spaces
from gym.utils import seeding

# Importing my own packages which may contain some bugs
from .Energy_Calc import *

# This class describes the formal environment which the reinforcement learning
# interacts with. It inherits some properties from the gym imported earlier
class Env(gym.Env):

    def __int__(self, datapath, modelpath: list, period = 6):

        # State Space variables
        self.vars = ['OAT', 'RH', 'AirFlow', 'CC_T',  'PH_T', 'SAT', 'TotalE']
        self.states = ['OAT', 'RH', 'AirFlow', 'PH_T', 'SAT']

        # Here we initialize the data driven model for evaluating energy
        # The weights and biases of the models are stored in a file
        self.precoolmodel = load(modelpath[0])
        self.recovheatmodel = load(modelpath[1])

        # data to be used for training and testing
        self.traindataset = read_pickle(datapath)
        self.traindataset = self.traindataset[self.vars]

        # parameters
        self.period = period
        self.numactions = 2

        # shape of data frame
        self.m, self.n = self.traindataset.shape

        # slicing data into train and test sequences
        self.slicepoint = 0.75
        self.traindatalimit = int(self.slicepoint*self.m)
        self.testdatalimit = self.m

        # getting 0:mean 1:std 2:min 3:max- maintain a dataframe as much
        # as possible
        self.Stats = self.traindataset.describe().iloc[[1, 2, 3, 7], :-1]

        # Windowed Stats: Assuming a window of 24 hours
        self.win_len = int(1440 / (5 * self.period))
        self.windowMean = self.traindataset.rolling(self.win_len, min_periods=1).mean()['OAT']
        self.windowMax = self.traindataset.rolling(self.win_len, min_periods=1).max()['OAT']
        self.windowMin = self.traindataset.rolling(self.win_len, min_periods=1).min()['OAT']

        # Standard requirements for interfacing with Keras-RL's code
        SpaceLB = [self.Stats.iloc[2, i] for i in range(self.n - self.numactions)]
        SpaceUB = [self.Stats.iloc[3, i] for i in range(self.n - self.numactions)]
        self.observation_space = spaces.Box(low=np.array(SpaceLB),
                                            high=np.array(SpaceUB),
                                            dtype=np.float32)
        self.action_space = spaces.Box(low=self.Stats.iloc[2, [-3, -2]].to_numpy(),
                                       high=self.Stats.iloc[3, [-3, -2]].to_numpy(),
                                       dtype=np.float32)
        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

        # counter: counts the current step number in an episode
        # episodelength: dictates number of steps in an episode
        # testing: whether we env is in testing phase or not
        # dataPtr: steps through the entire available data in a cycle- gets
        #           reset to 0 when entire trainData is used up
        self.counter = 0
        self.testing = False
        self.dataPtr = 0
        # setting episode length to 1 week since test duration is 1 week. Then we can have 4 episodes for training
        # 1 week = 10080 mins
        self.episodelength = int(10080 / (period * 5))  # eg 336 steps when period = 6 ie 30 mins interval

        '''Resetting the environment to its initial value'''
        self.S = self.traindataset.iloc[self.dataPtr, :-1]  # Added to the control action generated
        self.state = self.S[self.states].to_numpy().flatten()

        # Models needed for calcualting physics of the AHU
        self.energyCalc = Energy_Calc()
        # passing updated values before calculation
        self.energyCalc.updateVars(self.S)

    def testenv(self):
        self.testing = True
        self.dataPtr = self.traindatalimit

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, controlact):  # have to see model summary to figure out shape of controlAct

        # preheat set point
        pht_stp = controlact[0]
        # reheat set point
        rht_stp = controlact[1]
        # airflow rate
        airflow = self.S['AirFlow']
        # outside air temperature
        oat = self.S['OAT']
        # cooling coil exit temperature
        cc_t = self.S['CC_T']

        # Calculate the preheat energy consumption assuming
        # we attain the pht_stp from oat
        if oat<52:
            pht_energy = self.energyCalc.preheatenergy(airflow,
                                                       oat,
                                                       pht_stp)
            pht_temp = pht_stp
        else:
            pht_energy = 0
            pht_temp = self.S['PH_T']

        # calculate precool temp
        precooltemp = self.precoolmodel(pht_temp, airflow)

        # calculate cooling energy
        cooling_energy = self.energyCalc.CoolingEnergy(precooltemp, cc_t, airflow)

        # calculate recovheat temperature
        recovtemp = self.recovheatmodel(oat, pht_temp, cc_t, airflow)

        # calculate