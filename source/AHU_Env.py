import gym
from gym import spaces, logger
from gym.utils import seeding

from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.model_selection import train_test_split
import numpy as np
from pandas import *

#Importing my own packages which may contain some bugs
import Energy_Calc

from joblib import dump, load
import importlib

#need to run this if I make changes to Energy_Calc and other modules
importlib.reload(Energy_Calc)
from Energy_Calc import Energy_Calc


class Env(gym.Env):
    
    def __init__(self):
        #load total data
        totalData = read_pickle("./totalData.pkl")
        totalData.drop(['PH_T_energy','Cooling_energy'],axis=1,inplace=True)
        #Preparing the data for the Preheat=0; RelativeHumidity>60%; OAT>52F; Reheat>0; Cool>0
        #We are selecting these limits based on reccommendations from the engineering drawing charts of the AHU
        #We are interested especially where there is RH>60% and Ambient Temperature>52F
        totalData = totalData[totalData[2]>0]#removing negative fan flow values
        RL_data = totalData[totalData[5]==0]#PreHeat=0
        RL_data = RL_data[RL_data[12]>60]#RelativeHumidity>60%
        RL_data = RL_data[RL_data[0]>52]#OAT>52F
        RL_data = RL_data[RL_data[4]>0]#Reheat>0
        RL_data = RL_data[RL_data[3]>0]#Cool>0
        self.dfAnalysis = RL_data
        
        #load the HeatRecov, Reheat Output and the Discharge Temperature.Recovery Heat models
        #self.HeatRecov = load('HeatRecov.joblib') not needed here
        #self.ReheatOutput = load('Reheat_OP.joblib')#May not need this
        #self.DAT = load('DAT.joblib') #May not need this
        
        #mean and std values for the dataset under high temp and high humidity conditions
        self.Stats = self.dfAnalysis.describe().iloc[[1,2],:].values
        #The new scaled data
        #This is only used for learning the state space for the re-inforcement learning problem
        #Not in the actual energy calculation. The energy calculation cannot be scaled.
        tobeScaled = self.dfAnalysis.columns
        scaler= StandardScaler()
        self.scaledData = DataFrame(scaler.fit_transform(self.dfAnalysis))
        self.scaledData.columns = tobeScaled
        
        #Standard requirements for interfacing with Keras-RL's code
        #State: OAT, DAT, CC_T, ORH, DewPt, recovHeatTemp 
        #Actions: CC_STP,DAT_STP
        
        #This formulation of State and Axn is true nly when preheat is not operating
        #The secondary goal would be to adjust the RH of the outgoing air as close to 50%
        #Bounds are typical of the conditions for which the RL problem is being
        #designed
        scaledObsvSpaceLB = [-3,-2.5,-3,-3,-2.5,-4]
        scaledObsvSpaceUB = [3,2,4,3,1.6,6]
        self.observation_space = spaces.Box(low=np.array(scaledObsvSpaceLB),\
             high=np.array(scaledObsvSpaceUB),dtype=np.float32)
        #self.action_space = spaces.Box(low=np.array([51,65]),\
                                       #high=np.array([60,75]), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-1,-1]),\
                                       high=np.array([1,1]), dtype=np.float32)
        #The bounds are determined by looking at the histogram for DewPoint!! and DAT_STP
        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        
        #Reset state
        #Normalized Information
        self.counter = 0
        self.threshold = 4000
        self.s_t_n = self.scaledData.loc[self.counter,[0,1,10,12,13]].to_numpy()
        ccInit_in = 70
        #cc_in_nor = (ccInit_in-self.Stats[0,10])/self.Stats[1,10]
        recovInit_nor = (70-self.Stats[0,1])/self.Stats[1,1]
        self.s_t_n = np.append(self.s_t_n, recovInit_nor)
        #self.s_t_n = np.append(self.s_t_n, cc_in_nor)
        #Unnormalized Information
        self.un_index = self.dfAnalysis.index
        self.s_t = self.dfAnalysis.loc[self.un_index[self.counter],:]
        self.s_t['recovHeatTemp'] = 70
        self.s_t['recovCoolTemp'] = ccInit_in
        '''What do we initialize it with? data value right before the reheat operation starts? ie look it up from the data?'''
        #not needed in state the pressure
        #self.Pressure_t = self.dfAnalysis.loc[self.un_index[self.counter],[14]].to_numpy()[0]
        #not  needed in state but needs to persist in memory
        self.tempOP = []#Stores the calculated cooling and reheat outputs
        self.EnergyCons = []#Stores the calculated cooling and reheat outputs
        self.relHum_controlled = []#Stores the relHum obtained using RL
        self.relHum_original = []#Stores RL from original setpoint
        
        #Models needed for calcualting physics of the AHU
        self.energyCalc = Energy_Calc()
        #passing updated values before calculation
        self.energyCalc.updateVars(self.s_t)
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def boundsMapper(self,controlAct):
        #ToDO
        #cctspt would be mapped by dewpoint bounds from self.Stats variable/decided bound by meeting
        #dat stp would be mapped by it's own bounds from self.Stats variable/decided bound by meeting
        tanhdelta = 1 - (-1)
        lowth = [51,65]
        highth = [60,75]
        deltath =[25,10]
        
        CC_STP = lowth[0] + (controlAct[0,0]+1)*deltath[0]/tanhdelta
        DAT_STP = lowth[1] + (controlAct[0,1]+1)*deltath[1]/tanhdelta
        
        return [CC_STP,DAT_STP]
    
    def step(self,controlAct):
        #Here ControlAct is tanh scaled for convergence
        #We need a bound mapping function
        #See keras rl  on how to adjust the main code for two dimensional actions
        
        #We receive the control action at state s_t
        #Finding the actual set points
        a_t = self.boundsMapper(controlAct)
        
        #Calculating the temperature of the air exiting the recovery heat coil and entering the cooling coil
        #The data to be used is to be from the 
        #input at t-1: the flow rate(2), the temperature of the prht coil sensor(8)
        #output at t : temperature of the cooling coil(10) but only for training
        cc_in = self.energyCalc.CC_in()
        #norrmalize using the 
        cc_in_n = (cc_in-self.Stats[0,10])/self.Stats[1,10]
        
        #Calculating the temperature of air exiting the cooling coil and also temp of air entering reheat coil
        #Based on the cct_spt we evaluate the new CC_T and then recovHeatTemp #Accessing a tensor output?
        cc_t,recovHeatTemp = self.energyCalc.CooledAirTemp(a_t[0])
        #Normalized values
        cc_t_n = (cc_t-self.Stats[0,10])/self.Stats[1,10]
        recovHeatTemp_n = (recovHeatTemp-self.Stats[0,1])/self.Stats[1,1]#substituting meand and std of recovHeat with DAT
        #since std and mean of recovHeat are not available
            
        #Calculating the temperature of air exiting the reheat coil
        #based on action a_t
        dat = a_t[1]
        dat_n = (dat-self.Stats[0,1])/self.Stats[1,1]
        '''^^^^^^So, we have to assume that we can always achieve the desired setpoint and remove any model assumption about
        modeling the reheat output. So can we remove the acutalTemp?. NO! we need it to calculate the enthalpy change
        For the cooling energy calculation, we have to use the temp out of the recovery coil into the cooling coil'''
        
        #Before we calculate energy, we must update state to reflect the new energy based on 
        #the effects of the control actions
        #incease counter so the state t+1 can be properly created
        self.counter += 1
        
        #Normalized
        self.s_t_n = self.scaledData.loc[self.counter,[0,1,10,12]].to_numpy()
        #updating with actual state values at t+1
        self.s_t_n[1] = dat_n
        self.s_t_n[2] = cc_t_n
        self.s_t_n = np.append(self.s_t_n, recovHeatTemp_n)
                
        #Information passed to calculate energy
        self.s_t = self.dfAnalysis.loc[self.un_index[self.counter],:]#pandas series
        #updating with actual state values at t+1
        self.s_t[1] = dat
        self.s_t[10] = cc_t
        self.s_t[7] = a_t[0]
        self.s_t[11] = a_t[1]
        self.s_t['recovHeatTemp'] = recovHeatTemp
        self.s_t['recovCoolTemp'] = cc_in
        #passing updated values before energy calculation
        self.energyCalc.updateVars(self.s_t)
        
        
        #Now we calculate reheating the energy
        cooling_energy,org_cooling_energy = self.energyCalc.CoolingEnergy(53)
        #Reheat implicitly calculates the relative humidity exiting the coil
        #Now we calculate reheating the energy
        reheating_energy,org_reheating_energy = self.energyCalc.ReheatEnergy(73)
        #Reheat implicitly calculates the relative humidity exiting the coil
        
        #Storing the energyValues
        self.EnergyCons.append([reheating_energy,org_reheating_energy])
        
        #Calculate the reward
        #should we try to see the relative humidity?
        #For comparison, we need to have a database which has the original energy consumption
        '''Calculate energy consumption based on actual set point-done!
        Calcualte the relative humidity of the air--done'''
        reward = 0
        if (org_reheating_energy-reheating_energy) > 0:
            reward = reward + 0.5;
        if np.abs(0.5-self.energyCalc.RH_rheat_out) < 0.05:
            reward = reward + 1;
            
        self.relHum_controlled.append(self.energyCalc.RH_rheat_out)
        self.relHum_original.append(self.energyCalc.RH_rheat_org)
        
        done = False
        if self.counter>self.threshold:
            done=True
        
        return self.s_t_n,reward,done,{}
    
    def reset(self):
        
        #Reset state
        #Normalized Information
        self.counter = 0
        self.threshold = 4000
        self.s_t_n = self.scaledData.loc[self.counter,[0,1,10,12,13]].to_numpy()
        ccInit_in = 70
        cc_in_n = (ccInit_in-self.Stats[0,10])/self.Stats[1,10]
        recovInit_nor = (70-self.Stats[0,1])/self.Stats[1,1]
        self.s_t_n = np.append(self.s_t_n, recovInit_nor)
        #Unnormalized Information
        self.un_index = self.dfAnalysis.index
        self.s_t = self.dfAnalysis.loc[self.un_index[self.counter],:]
        self.s_t['recovHeatTemp'] = 70
        '''What do we initialize it with? data value right before the reheat operation starts? ie look it up from the data?'''
        #not needed in state the pressure
        #self.Pressure_t = self.dfAnalysis.loc[self.un_index[self.counter],[14]].to_numpy()[0]
        #not  needed in state but needs to persist in memory
        self.tempOP = []#Stores the calculated cooling and reheat outputs
        self.EnergyCons = []#Stores the calculated cooling and reheat outputs
        self.relHum_controlled = []#Stores the relHum obtained using RL
        self.relHum_original = []#Stores RL from original setpoint
        
        return self.s_t_n
        
#To Do: Implement the appropriate data format which includes the recovHeatTemp in column numbered 15 ie first download the data
#Build the GBM models for the reheat output, discharge air temperature
#Find out a suitable value for the threshold
#Figure out the reward function
#Whether to record self.tempOP or not
#Check whether the reset state method is correct?
