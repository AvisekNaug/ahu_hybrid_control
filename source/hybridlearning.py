# import  modules
from . helperfunctions import *
from . dataGenerator import *
from . predictionmodel import *

# initially used to process raw data
def customdf_ahu(ahupath, ahu, savepath, metasysdata, limit=0.1):
    # read the relevant data
    data_path = [ahupath+'/BdXdata/', ahupath+'/Solcastdata/', ahupath+'/valvedata/']
    date_column_name = ['Date', 'PeriodEnd', 'Time']
    date_format = ['%m/%d/%Y %H:%M', '%Y-%m-%dT%H:%M:%SZ', None]
    outlier_names = [['AirFlow', 'CC_T', 'OAT', 'PH_T.', 'SAT'], [], []]
    time_offsets = [0, 0, 0]
    # Create the data frame
    df = createdataframe(data_path, date_column_name, date_format, outlier_names, time_offsets,
                         metasysdata, limit=limit)
    # drop solcast air temp and dew point
    df.drop(columns=['AirTemp', 'DewpointTemp'], inplace=True)
    # rename columns: maintain order of variables correctly
    namedict = {'OAT': 'OAT',
                'AirTemp': 'OAT',
                'SAT': 'SAT',
                'SAT_STP': 'SAT_STP',
                'AirFlow': 'AirFlow',
                'PHT_STP': 'PHT_STP',
                'CCT_STP': 'CCT_STP',
                'PH_T.': 'PH_T',
                'CC_T': 'CC_T',
                'RelativeHumidity': 'RH',
                'AHU_2.preheatOutput': 'P_OP',
                'AHU_2.coolOutput': 'C_OP',
                'AHU_2.heatOutput': 'R_OP',
                'Preheat Output.Preheat Output.Trend - Present Value ()': 'P_OP',
                'Chilled Water Valve Ouptut.Chilled Water Valve Ouptut.Trend - Present Value ()': 'C_OP',
                'Reheat Output.Reheat Output.Trend - Present Value ()': 'R_OP',
                'DewpointTemp': 'DP'}
    # df.columns = ['OAT', 'SAT', 'SAT_STP', 'AirFlow', 'PHT_STP', 'CCT_STP',
    #               'PH_T', 'CC_T', 'RH', 'P_OP', 'C_OP', 'R_OP']
    df.columns = [namedict[i]+ahu for i in df.columns]

    # save the data frame
    dfsave(df, savepath)

# params
ahu1 = 'ahu1'
ahu2 = 'ahu2'
metasysdataahu1 = [False, False, True]
metasysdataahu2 = [False, False, False]

# create custom dataframes
customdf_ahu('./data/ahu1', ahu1, 'hybrid_data_ahu1.pkl', metasysdataahu1, limit=0.1)
customdf_ahu('./data/ahu2', ahu2, 'hybrid_data_ahu2.pkl', metasysdataahu2, limit=0.5)

# read the dataframe from stored pickled data
df1 = read_pickle('hybrid_data_ahu1.pkl')
df2 = read_pickle('hybrid_data_ahu2.pkl')

# remove old results files
removeoldresults('ResultsAHU1')
removeoldresults('ResultsAHU2')

# learn function for estimating recovery heat air temperature for AHU 1
X_train, X_test, y_train, y_test = recovheatdata(df1, ahu1)
model = GBR_model(modeltype='Recovery Heat Temp', period=1, savepath='ResultsAHU1')
model.trainmodel(X_train, X_test, y_train, y_test, savemodel=True)

# learn function for estimating pre cool air temperature for AHU 1
X_train, X_test, y_train, y_test = precooldata(df1, ahu1)
model = GBR_model(modeltype='PreCool Temp', period=1, savepath='ResultsAHU1')
model.trainmodel(X_train, X_test, y_train, y_test, savemodel=True)

# learn function for estimating recovery heat air temperature for AHU 2
X_train, X_test, y_train, y_test = recovheatdata(df2, ahu2)
model = GBR_model(modeltype='Recovery Heat Temp', period=1, savepath='ResultsAHU2')
model.trainmodel(X_train, X_test, y_train, y_test, savemodel=True)

# learn function for estimating pre cool air temperature for AHU 2
X_train, X_test, y_train, y_test = precooldata(df2, ahu2)
model = GBR_model(modeltype='PreCool Temp', period=1, savepath='ResultsAHU2')
model.trainmodel(X_train, X_test, y_train, y_test, savemodel=True)

# learn the necessary driven models from the provided data
def customdf_condensor(savepath='condensordata.pkl', metasysdata=[False], limit=0.1):
    # read the relevant data
    data_path = ['./data/condensor/']
    date_column_name = ['Date']
    date_format = ['%m/%d/%Y %H:%M']
    outlier_names = [['Alumni_Hall_Cond_Loop_S_T.value',
                     'Alumni_Hall_Cond_Loop_R_T.value']]
    time_offsets = [0]
    # Create the data frame
    df = createdataframe(data_path, date_column_name, date_format, outlier_names, time_offsets,
                         metasysdata, limit=limit)
    # drop solcast air temp and dew point
    df.drop(columns=['Condensor_Water_Pump.pumpVfdPercent',
                     'Secondary_Chilled_Water_Pump.pumpVfdPercent'], inplace=True)
    # rename columns: maintain order of variables correctly
    namedict = {'OAT': 'OAT',
                'AirTemp': 'OAT',
                'SAT': 'SAT',
                'SAT_STP': 'SAT_STP',
                'AirFlow': 'AirFlow',
                'PHT_STP': 'PHT_STP',
                'CCT_STP': 'CCT_STP',
                'PH_T.': 'PH_T',
                'CC_T': 'CC_T',
                'RelativeHumidity': 'RH',
                'AHU_2.preheatOutput': 'P_OP',
                'AHU_2.coolOutput': 'C_OP',
                'AHU_2.heatOutput': 'R_OP',
                'Preheat Output.Preheat Output.Trend - Present Value ()': 'P_OP',
                'Chilled Water Valve Ouptut.Chilled Water Valve Ouptut.Trend - Present Value ()': 'C_OP',
                'Reheat Output.Reheat Output.Trend - Present Value ()': 'R_OP',
                'DewpointTemp': 'DP',
                'Alumni_Hall_Cond_Loop_R_T.value': 'CondRT',
                'Alumni_Hall_Cond_Loop_S_T.value': 'CondST',
                'Alumni_Hall_SCHW1_DP.value': 'Ahu1DP',
                'Alumni_Hall_SCHW2_DP.value': 'Ahu2DP',
                'Alumni_Hall_CU_DP.value': 'CuDP',
                'Alumni_Hall_PCHW_Flow.value': 'PchwFlow'}
    df.columns = [namedict[i] for i in df.columns]

    # save the data frame
    dfsave(df, savepath)
# create custom dataframes
customdf_condensor()
df3 = read_pickle('condensordata.pkl')

# setup environment to learn the appropriate control method
# this section will be written in a separate script
