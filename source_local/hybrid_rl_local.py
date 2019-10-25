from pandas.core.common import SettingWithCopyWarning
from source_local.AHU1_env_local import *
from source_local.agent import *
from source_local.rl_perf_plot_local import *

import os
import warnings
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

# parameters for the problem
period = 1  # ie period*5 minutes eg 12*5 60 minutes
datapath = '../traindata/hybrid_data_ahu1.pkl'
modelpath = ['../ResultsAHU1/PreCool_Temp_GBR_model_2000_estimators.joblib',
             '../ResultsAHU1/Recovery_Heat Temp_GBR_model_2000_estimators.joblib']
dest='agent_weights.h5f'


# Remove RL training log files
rllogs = '../RL_data'
# results save location
rllogs_local = '../RL_data_local'


# remove previous files
files = os.listdir('../RL_data_local/')
for f in files:
    os.remove(rllogs_local + '/' + f)

files = os.listdir('../RL_data/')
for f in files:
    os.remove(rllogs_local + '/' + f)

files = os.listdir("../td3_hvac_tensorboard/")
for f in files:
    os.remove(rllogs_local + '/' + f)


# Create the HVAC environment
env = Env(datapath, modelpath, period=period)


# Instantiating the agent for learning the control policy
agent = get_agent(env, rllogs)


# train the agent
train_agent(agent, rllogs, steps=10000)


# do testing
test_perf_log = test_agent(agent, env, rllogs_local)

# save performance metrics
rl_perf_save(test_perf_log, rllogs_local)

# energy comparison
rl_energy_compare('../RL_data_local/totalE_hist.txt',
                  '../RL_data_local/oat.txt',
                  '../RL_data_local/totalE.txt',
                  '../RL_data_local/', period=1)

# relative humidity comparison
relhumplot('../RL_data_local/rht_out_rh.txt',
           '../RL_data_local/oat.txt',
           '../RL_data_local/rht_out_rh_hist.txt',
           '../RL_data_local/', period=1)

# combined plot
oat_vs_control('../RL_data_local/splot.txt',
               '../RL_data_local/oat.txt',
               '../RL_data_local/', period=1)

plot_results('../RL_data/')
