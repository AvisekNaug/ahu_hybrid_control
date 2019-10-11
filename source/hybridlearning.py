# import  modules
from . helperfunctions import *
from . dataGenerator import *
from . predictionmodel import *

# initially used to process raw data
def customdf_ahu(ahupath, ahu, savepath, metasysdata):
    # read the relevant data
    data_path = [ahupath+'/BdXdata/', ahupath+'/Solcastdata/', ahupath+'/valvedata/']
    date_column_name = ['Date', 'PeriodEnd', 'Time']
    date_format = ['%m/%d/%Y %H:%M', '%Y-%m-%dT%H:%M:%SZ', None]
    outlier_names = [['AirFlow', 'CCT_STP', 'CC_T', 'OAT', 'PHT_STP', 'PH_T.'], [], []]
    time_offsets = [0, 0, 0]
    # Create the data frame
    df = createdataframe(data_path, date_column_name, date_format, outlier_names, time_offsets, metasysdata)
    # drop solcast air temp and dew point
    df.drop(columns=['AirTemp', 'DewpointTemp'], inplace=True)
    # rename columns: maintain order of variables correctly
    df.columns = ['OAT', 'SAT', 'SAT_STP', 'AirFlow', 'PHT_STP', 'CCT_STP',
                  'PH_T', 'CC_T', 'RH', 'P_OP', 'C_OP', 'R_OP']
    df.columns = [i+ahu for i in df.columns]

    # save the data frame
    dfsave(df, savepath)

# params
ahu1 = 'ahu1'
ahu2 = 'ahu2'
metasysdataahu1 = [False, False, True]
metasysdataahu2 = [False, False, False]

# create custom dataframes
customdf_ahu('./data/ahu1', ahu1, 'hybrid_data_ahu1.pkl', metasysdataahu1)
customdf_ahu('./data/ahu2', ahu2, 'hybrid_data_ahu2.pkl', metasysdataahu2)

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

# setup environment to learn the appropriate control method
