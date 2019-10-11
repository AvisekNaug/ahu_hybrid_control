# import modules
import numpy as np
from sklearn.model_selection import train_test_split

# generate training data for the hybrid control

# data for recovery heat temperature prediction
def recovheatdata(df,ahu):
    # input: oat(since the recovery jacket gets heat from that section)
    #        pht(temperature of air after passing through preheat coil)
    #        cct(that is the temperature at which air enters the recov heat section)
    #        airflow(rate at which air flows modulates rate of heat exchange)
    # output: dat(since reheat is off it is not changing the temperature)

    # select data where reheat output is 0
    df_noreheat = df[df['R_OP'+ahu]==0]

    # lag
    lag = -1

    # create data
    X = df_noreheat[['OAT'+ahu, 'PH_T'+ahu, 'CC_T'+ahu, 'AirFlow'+ahu]].to_numpy()[:lag, :]
    y = df_noreheat[['SAT'+ahu]].to_numpy()[-lag:].flatten()

    return train_test_split(X, y, test_size=0.25)

# data for pre cooled temperature prediction
def precooldata(df, ahu):
    # input: pht(temperature of air after passing through preheat coil)
    #        airflow(rate at which air flows modulates rate of heat exchange)
    # output: cct(since cooling is off it is not changing the temperature)

    # select data where reheat output is 0
    df_noreheat = df[df['C_OP'+ahu]==0]

    # lag
    lag = -1

    # create data
    X = df_noreheat[['PH_T'+ahu, 'AirFlow'+ahu]].to_numpy()[:lag, :]
    y = df_noreheat[['CC_T'+ahu]].to_numpy()[-lag:].flatten()

    return train_test_split(X, y, test_size=0.25)
