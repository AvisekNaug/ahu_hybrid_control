from pandas.core.common import SettingWithCopyWarning
from AHU1_env import *
from agent import *
from rl_perf_plot import *
# from source.dataGenerator import *
# from source.helperfunctions import *
import os
import warnings
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

# parameters for the problem
period = 1  # ie period*5 minutes eg 12*5 60 minutes
datapath = '../traindata/hybrid_data_ahu1.pkl'
modelpath = ['../ResultsAHU1/PreCool_Temp_GBR_model_2000_estimators.joblib',
             '../ResultsAHU1/Recovery_Heat Temp_GBR_model_2000_estimators.joblib']
time_steps = 1  # int(120/(period*5))
num_steps = 806400  # rl train steps 1680 67200 117600 33600 100800

# Remove RL training log file infos
rllogs = '../RL_data'
try:
    os.mkdir(rllogs)
except FileExistsError:
    files = os.listdir(rllogs+'/')
    for f in files:
        os.remove(rllogs + '/' + f)

# create the HVAC environment
env = Env(datapath, modelpath, period=period)

# Instantiating the agent for learning the control policy
agent = get_agent(env)

# do training
train_metrics = train_agent(agent, env, steps=num_steps, dest=rllogs + '/' + 'agent_weights.h5f')

# save training metrics
rl_reward_save(train_metrics, rllogs= rllogs)

# do testing
env.testenv()
test_perf_log = test_agent(agent, env, weights=rllogs + '/' + 'agent_weights.h5f')

# save performance metrics
rl_perf_save(test_perf_log, rllogs)

# plot the results
# episode wise reward
rl_reward_plot('../RL_data/Cumulative Episode Rewards.txt',
               '../RL_data/')
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
