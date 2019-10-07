# import  modules
from helperfunctions import *

# read the relevant data
data_path = ['./data/BdXdata/', './data/Solcastdata/','./data/valvedata/']
date_column_name = ['Date', 'PeriodEnd','Time']
date_format = ['%m/%d/%Y %H:%M', '%Y-%m-%dT%H:%M:%SZ', None]
outlier_names = [['AirFlow', 'CCT_STP', 'CC_T', 'OAT', 'PHT_STP', 'PH_T.'], [], []]
time_offsets = [0, 0, 0]
metasys = [False, False, True]

# Create the data frame
df = createdataframe(data_path, date_column_name, date_format, outlier_names, time_offsets, metasys)
# drop solcast air temp and dew point
df.drop(columns=['AirTemp', 'DewpointTemp'], inplace=True)
# rename columns
df.columns = ['AirFlow', 'CCT_STP', 'CC_T', 'OAT', 'PHT_STP', 'PH_T.',
              'SAT', 'SAT_STP', 'RH', 'C_OP', 'P_OP', 'R_OP']

# learn function for estimating intermediate temperature

# learn the necessary driven models from the provided data

# setup environment to learn the appropriate control method











# data inspection
def datainspect(df):
    from matplotlib import pyplot as plt
    plt.rcParams["figure.figsize"] = (20,30)
    fig,ax = plt.subplots(df.shape[1],1)
    for i,j in zip(df.columns,range(df.shape[1])):
        df.plot(y=[i],ax=ax[j],style=['b--'])