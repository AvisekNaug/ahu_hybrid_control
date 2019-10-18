# importing the modules for reading the data and pre-processing it
from pandas import *
from joblib import dump, load

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

    def __int__(self, datapath, period = 6):

        # State Space variables
        self.vars = ['OAT', 'RH', 'AirFlow', 'CC_T',  'PH_T', 'SAT', 'TotalE']
        self.states = ['OAT', 'RH', 'AirFlow', 'PH_T', 'SAT']

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
        # self.win_len = int(1440 / (5 * self.period))
        # self.windowMean = self.traindataset.rolling(self.win_len, min_periods=1).mean()['OAT']
        # self.windowMax = self.traindataset.rolling(self.win_len, min_periods=1).max()['OAT']
        # self.windowMin = self.traindataset.rolling(self.win_len, min_periods=1).min()['OAT']

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
        self.S = self.traindataset.iloc[self.dataPtr, :-1]
        self.state = self.S[self.states]

        # Models needed for calcualting physics of the AHU
        self.energyCalc = Energy_Calc()

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
        airflow = self.state['AirFlow']
        # outside air temperature
        oat = self.state['OAT']
        # cooling coil exit temperature
        cc_t = self.state['CC_T']
        # outside relative humidity
        orh = self.state['RH']/100

        # Calculate the preheat energy consumption assuming
        # we attain the pht_stp from oat
        if oat<=52:

            # resultant preheat output temperature
            ph_temp = pht_stp

            # calculate preheat energy and resultant relative humidity
            pht_energy, pht_out_rh, pht_energy_hist, pht_out_rh_hist = \
                self.energyCalc.preheatenergy(airflow, oat, ph_temp, orh)

            # calculate pre cool temp and resultant relative humidity
            precooltemp, pc_out_rh, precooltemp_hist, pc_out_rh_hist = \
                self.energyCalc.precooltemp(ph_temp, airflow, pht_out_rh, pht_out_rh_hist)

            # calculate cooling energy and resultant relative humidity
            cooling_energy, cc_out_rh, cooling_energy_hist, cc_out_rh_hist = 0, pc_out_rh, 0, pc_out_rh_hist

            # calculate recovheat temperature and resultant relative humidity
            recovtemp, rec_outrh = self.energyCalc.recovheattemp(oat, ph_temp, cc_t,
                                                                 airflow, cc_out_rh, cc_out_rh_hist)

            # calculate reheat energy consumption and resultant relative humidity
            reheat_energy, sat_outrh, sat = 0, rec_outrh, recovtemp


        else:

            # resultant preheat output temperature
            ph_temp = 75

            # calculate preheat energy and resultant relative humidity
            pht_energy, pht_out_rh, pht_energy_hist, pht_out_rh_hist = 0, orh, 0, orh

            # calculate pre cool temp and resultant relative humidity
            precooltemp, pc_out_rh, precooltemp_hist, pc_out_rh_hist =\
                self.energyCalc.precooltemp(ph_temp, airflow, pht_out_rh, pht_out_rh_hist)

            # calculate cooling energy and resultant relative humidity
            cooling_energy, cc_out_rh, cooling_energy_hist, cc_out_rh_hist = \
                self.energyCalc.coolingenergy(precooltemp, cc_t, airflow,
                                              pc_out_rh, pc_out_rh_hist, precooltemp_hist)

            # calculate recovheat temperature and resultant relative humidity
            recovtemp, rec_outrh = self.energyCalc.recovheattemp(oat, ph_temp, cc_t,
                                                                 airflow, cc_out_rh, cc_out_rh_hist)

            # supply air temperature
            sat = rht_stp

            # calculate reheat energy consumption and resultant relative humidity
            reheat_energy, outrh = self.energyCalc.ReheatEnergy(recovtemp, sat, airflow, rec_outrh)

        # calculate reward:
        reward = -pht_energy -cooling_energy -reheat_energy

        # move ahead in time
        self.dataPtr += 1
        self.counter += 1

        # adjust proper indexing of sequential train and test data
        if not self.testing:
            if self.dataPtr > self.traindatalimit - 1:
                self.dataPtr = 0
        else:
            if self.dataPtr > self.testdatalimit - 1:
                self.dataPtr = self.traindatalimit
        # see if episode has ended
        if self.counter>self.episodelength-1:
            done=True

        # step to the next state
        self.S = self.traindataset.iloc[self.dataPtr, :-1]
        self.state = self.S[self.states]

        # change states based on actions
        self.state['PHT'] = ph_temp
        self.state['SAT'] = sat

    def reset(self):
        self.S = self.traindataset.iloc[self.dataPtr, :-1]
        self.state = self.S[self.states]
        self.counter = 0
        self.steps_beyond_done = None
        return self.state
