from pandas.core.common import SettingWithCopyWarning
from source.AHU1_env_local import *
from source.agent import *
from source.rl_perf_plot_local import *
import os
import warnings
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

# parameters for the problem
period = 1  # ie period*5 minutes eg 12*5 60 minutes
datapath = '../traindata/hybrid_data_ahu1.pkl'
modelpath = ['../ResultsAHU1/PreCool_Temp_GBR_model_2000_estimators.joblib',
             '../ResultsAHU1/Recovery_Heat Temp_GBR_model_2000_estimators.joblib']

# Remove RL training log file infos
rllogs = '../RL_data'
# results save location
rllogs_local = '../RL_data_local'
files = os.listdir(rllogs_local+'/')
for f in files:
    os.remove(rllogs_local + '/' + f)

# create the HVAC environment
env = Env(datapath, modelpath, period=period)

# Instantiating the agent for learning the control policy
agent = get_agent(env)

# do testing
env.testenv()
env.dataPtr = 40000
test_perf_log = test_agent(agent, env, weights=rllogs + '/' + 'agent_weights.h5f')

# save performance metrics
rl_perf_save(test_perf_log, rllogs_local)
# from source.rl_perf_plot_local import *
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

