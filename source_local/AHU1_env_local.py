# importing the modules for reading the data and pre-processing it
from pandas import *
from joblib import dump, load

# importing the modules for setting up the environment from which the
# algorithm learns the control policy to be implemented
import gym
from gym import spaces
from gym.utils import seeding

# Importing my own packages which may contain some bugs
from source.Energy_Calc_local import *


# This class describes the formal environment which the reinforcement learning
# interacts with. It inherits some properties from the gym imported earlier
class Env(gym.Env):

    def __init__(self, datapath, modelpath, period=1):

        # State Space variables
        self.vars = ['OATahu1', 'RHahu1', 'AirFlowahu1', 'PH_Tahu1', 'SATahu1', 'CC_Tahu1']
        self.states = ['OATahu1', 'RHahu1', 'AirFlowahu1', 'PH_Tahu1', 'SATahu1']
        # parameters
        self.period = period
        self.numactions = 2

        # Class needed for calculating physics of the AHU
        self.energyCalc = energy_calc(modelpath)

        # data to be used for training and testing
        self.traindataset = read_pickle(datapath)
        # select only the variables of interest
        self.traindataset = self.traindataset[self.vars]

        # shape of data frame
        self.m, _ = self.traindataset.shape
        self.n = len(self.states)

        # slicing data into train and test sequences
        self.slicepoint = 0.75
        self.traindatalimit = int(self.slicepoint*self.m)
        self.testdatalimit = self.m

        # getting 0:mean 1:std 2:min 3:max- maintain a dataframe as much
        # as possible
        self.Stats = self.traindataset.describe().iloc[[1, 2, 3, 7], :]
        self.Stats = self.Stats.loc[:, self.states]

        # Standard requirements for interfacing with Keras-RL's code
        SpaceLB = [self.Stats.iloc[2, i] for i in range(self.n)]
        SpaceUB = [self.Stats.iloc[3, i] for i in range(self.n)]
        self.observation_space = spaces.Box(low=np.array(SpaceLB),
                                            high=np.array(SpaceUB),
                                            dtype=np.float32)
        # self.action_space = spaces.Box(low=self.Stats.iloc[2, [-3, -2]].to_numpy(),
        #                                high=self.Stats.iloc[3, [-3, -2]].to_numpy(),
        #                                dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([65, 55]),
                                       high=np.array([75, 75]),
                                       dtype=np.float32)
        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

        # counter: counts the current step number in an episode
        # episode length: dictates number of steps in an episode
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
        self.S = self.traindataset.iloc[[self.dataPtr]]
        self.state = self.S[self.states]

        # previous action values to make small changes
        self.prev_pht = self.state['PH_Tahu1'].iloc[0]
        self.prev_rht = self.state['SATahu1'].iloc[0]

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
        # outside air temperature
        oat = self.state['OATahu1'].iloc[0]
        # airflow rate
        airflow = self.state['AirFlowahu1'].iloc[0]
        # cooling coil exit temperature
        cc_t = self.S['CC_Tahu1'].iloc[0]
        # outside relative humidity
        orh = self.state['RHahu1'].iloc[0]/100
        # ph_temp = None

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
            cooling_energy, cc_out_rh, cooling_energy_hist, cc_out_rh_hist = \
                0, pc_out_rh, 0, pc_out_rh_hist

            # calculate recovheat temperature and resultant relative humidity
            recovtemp, recov_out_rh, recovtemp_hist, recov_out_rh_hist = \
                self.energyCalc.recovheattemp(oat, ph_temp, cc_t,
                                              airflow, cc_out_rh, cc_out_rh_hist)

            # calculate reheat energy consumption and resultant relative humidity
            rht_energy, rht_out_rh, sat, rht_energy_hist, rht_out_rh_hist, sat_hist = \
                0, recov_out_rh, recovtemp, \
                0, recov_out_rh_hist, recovtemp_hist

            # for plotting
            splot = ph_temp

        else:
            # resultant preheat output temperature
            ph_temp = oat

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
            recovtemp, recov_out_rh, recovtemp_hist, recov_out_rh_hist = \
                self.energyCalc.recovheattemp(oat, ph_temp, cc_t,
                                              airflow, cc_out_rh, cc_out_rh_hist)

            # calculate reheat energy consumption and resultant relative humidity
            rht_energy, rht_out_rh, sat, rht_energy_hist, rht_out_rh_hist, sat_hist = \
                self.energyCalc.reheatenergy(airflow, recovtemp, recov_out_rh,
                                             recovtemp_hist, recov_out_rh_hist,
                                             rht_stp, self.S['SATahu1'].iloc[0])

            # for plotting
            splot = sat

        penalty = max(np.abs(self.prev_pht - ph_temp)-2, 0) + max(np.abs(self.prev_rht - sat)-2, 0)

        # calculate reward:
        t1 = 1 if pht_energy_hist - pht_energy > 0 else 0
        t2 = 1 if cooling_energy_hist - cooling_energy > 0 else 0
        t3 = 1 if rht_energy_hist - rht_energy > 0 else 0
        t4 = 1 if np.abs(self.prev_pht - ph_temp) < 2 else -0.2
        t5 = 1 if np.abs(self.prev_rht - sat) < 2 else -0.2
        reward = t1 + t2 + t3 + t4 + t5
        # reward = -float(pht_energy) -float(cooling_energy) -float(rht_energy) -float(penalty)

        # previous action values to make small changes
        self.prev_pht = ph_temp
        self.prev_rht = sat

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
        done = False
        if self.counter>self.episodelength-1:
            done = True

        # step to the next state
        self.S = self.traindataset.iloc[[self.dataPtr]]
        self.state = self.S[self.states]

        # change states based on actions
        self.state['PH_Tahu1'].iloc[0] = ph_temp
        self.state['SATahu1'].iloc[0] = sat

        if self.testing:
            accumulated_info = {
                'splot': float(splot),
                'oat': float(oat),
                'pht_stp': float(ph_temp),
                'pht_stp_hist': float(73),
                'rht_stp': float(sat),
                'rht_stp_hist': float(sat_hist),
                'rht_out_rh': float(rht_out_rh),
                'rht_out_rh_hist': float(rht_out_rh_hist),
                'pht_energy': float(pht_energy),
                'pht_energy_hits': float(pht_energy_hist),
                'cooling_energy': float(cooling_energy),
                'cooling_energy_hist': float(cooling_energy_hist),
                'rht_energy': float(rht_energy),
                'rht_energy_hist': float(rht_energy_hist),
                'totalE': float(pht_energy+cooling_energy+rht_energy),
                'totalE_hist': float(pht_energy_hist + cooling_energy_hist + rht_energy_hist)
            }
        else:
            accumulated_info = {}

        return self.state.to_numpy().flatten(), reward, done, accumulated_info

    def reset(self):
        self.S = self.traindataset.iloc[[self.dataPtr]]
        self.state = self.S[self.states]
        self.counter = 0
        self.steps_beyond_done = None
        return self.state.to_numpy().flatten()
