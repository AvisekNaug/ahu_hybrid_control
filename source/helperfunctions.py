import glob
from pandas import *
from scipy import stats
import os
import numpy as np


# Helper methods for the data collection
def fileReader(pathtofile, dateheading, format='%m/%d/%Y %H:%M', offset=0):
    """
    reads files in Bdx format and returns a list of data frames with parsed time
    :param pathtofile: type str; the folder or path from which we read individula .csv or .excel files
    :param dateheading: type str; the column name for date varies, so it is required
    :param format: format string for datetime parsing
    :return: list of dataframes
    """
    # Read the files
    datadirectory = pathtofile
    flist = glob.glob(datadirectory + '*')
    dlist = []
    for filename in flist:
        if filename.endswith('.csv'):
            df = read_csv(filename)
        if filename.endswith('.xlsx'):
            df = read_excel(filename)

        # Parsing the Date column
        try:
            df.insert(loc=0, column='Dates',
                      value=to_datetime(df[dateheading],
                                        format=format) + DateOffset(hours=offset))
        except ValueError:
            try:
                df.insert(loc=0, column='Dates',
                          value=to_datetime(df[dateheading],
                                            format='%m/%d/%Y %H:%M') + DateOffset(hours=offset))
            except ValueError:
                df.insert(loc=0, column='Dates',
                          value=to_datetime(df[dateheading],
                                            format='%Y-%m-%d %H:%M:%S') + DateOffset(hours=offset))

        df.drop(dateheading, axis=1, inplace=True)  # Drop original Time column

        # Add df to the dlist
        dlist.append(df)

    return dlist

def smoothcolumns(df, column_names=None, windowlength=4):
    """
    Smoothes the dataframe columns
    :param df: the input datafrme
    :param column_names: list of column names to be smoothed
    :param windowlength: window over which smoothing operation has to be done
    :return: smoothed data frame
    """
    if column_names is None:
        for i in df.columns:
            df[i]=df[i].rolling(windowlength, min_periods=4).mean()
        df.dropna(axis=0, how='any', inplace=True)

        return df

    else:
        if isinstance(column_names,list):
            for i in column_names:
                df[i] = df[i].rolling(windowlength, min_periods=4).mean()
            df.dropna(axis=0, how='any', inplace=True)

            return df
            
        else:
            raise TypeError("column_names must be a list")
    

def merge_df_rows(dlist):
    """
    Merge rows of dataframes sharing same columns but different time points
    Always Call merge_df_rows before calling merge_df_columns as time has not been set as
    index yet
    :param dlist: list of dataframes to be added along rows
    :return: dataframe
    """

    # Create Dataframe from the dlist files
    dframe = concat(dlist, axis=0, join='outer', sort=True)

    # Sort the df based on the datetime index
    dframe.sort_values(by='Dates', inplace=True)

    # Setting Dates as the dataframe index
    dframe.set_index(['Dates'], drop=True, inplace=True)

    # Dropiing duplicated time points that may exist in the data
    dframe = dframe[~dframe.index.duplicated()]

    return dframe


def merge_df_columns(dlist):
    """
    Merge dataframes  sharing same rows but different columns
    :param dlist: list of dataframes to be along column axis
    :return: concatenated dataframe
    """
    dframe = concat(dlist, axis=1, join='outer', sort=True)

    return dframe


def removeOutliers(df, columnnames, z_thresh=3):
    """
    remove outliers beyond 3 standard deviations in the data
    :param df:dataframe
    :param columnnames: listes of column names to check for outliers
    :param z_thresh: remove points beyond threshold*sima std deviations
    :return: cleaned dateframe
    """
    if columnnames == []:
        columnnames = df.columns
    for column_name in columnnames:
        # Constrains will contain `True` or `False` depending on if it is a value below the threshold.
        constraints = stats.zscore(df[column_name]) < z_thresh
        # Drop (inplace) values set to be rejected
        df.drop(df.index[~constraints], inplace=True)

    return df


def df_sample(df, period=12):
    """
    resamples dataframe at "period" 5 minute time points
    :param df:
    :param period: number of 5 min time points
    :return: sampled dataframe
    """
    timegap = period * 5
    return df[df.index.minute % timegap == 0]


def sparseCols(df, limit=0.2):
    """
    prints column names with more than "limit" fraction of data missing
    :param df:
    :param limit:
    :return:
    """
    print(df.columns[df.isnull().mean() > limit])


def removeSparseCols(df, limit=0.2):
    """
    Retrun dataframe with the sparse cols removed
    remove cols which are missing more than limit fraction of the data
    :param df:
    :param limit:
    :return: df
    """
    return df[df.columns[df.isnull().mean() < limit]]


def keepcols(df, col_list=None):
    """
    removes redundant cols from some dateframes
    :param df: dataframe
    :param col_list: list of cols to be retained
    :return: reduced dataframe
    """
    # return df.drop(labels=col_list,axis=1)
    if col_list is None:
        return df
    return df[col_list]


def droprows(df):
    return df.dropna(axis=0, how='any')


def dfsave(df, filename):
    """
    :param df: DataFrame, dateframe to pickle
    :param filename: str, filename to pickle to
    :return: None
    """
    return df.to_pickle(filename)


def rowAverage(df, columnName):
    """
    Computes Average across each row and populates it into a data frame
    :param df: DataFrame
    :param columnName: str name of the column
    :return: DataFrame with columnName
    """
    dfavg = DataFrame(index=df.index)
    dfavg[columnName] = df.mean(axis=1)

    return dfavg

# standard preprocessing on the dataframe
def createdataframe(datapath: list, datecolumn_name: list, dateformat: list, outliers: list,
                    time_offset: list, period: int = 1, limit: float = 0.1, **kwargs):

    assert len(datapath)!=0, "No datapaths provided"

    dflist = []
    for i, j, k, l, p in zip(datapath, datecolumn_name, dateformat, time_offset, outliers):

        # check whether it is a string
        assert type(i)==str, "Data path is not a string. It is of type {}.".format(type(i))
        # check whether the directory exists
        assert os.path.exists(i), "Directory {} does not exist.".format(i)

        # read the excel file into a dataframe
        df = fileReader(i, j, format=k, offset = l)
        # merge list of dfs along the row axis
        df = merge_df_rows(df)
        # remove outliers
        df = removeOutliers(df, p)
        # remove extremley sparse values
        df = removeSparseCols(df, limit=limit)
        # remove missing rows
        df = droprows(df)
        # # keep columns
        # if q:
        #     df = df.filter(items=q)
        # # rename columns
        # if r:
        #     df.columns = r
        # add df to list
        dflist.append(df)

    # merge dataframe lists into a single data frame
    df = merge_df_columns(dflist)
    # drop rows with missing values
    df = droprows(df)
    # # rearrange columns
    # df = df.reindex(columns=desired_order)

    return df