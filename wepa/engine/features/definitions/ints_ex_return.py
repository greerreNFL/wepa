import pandas as pd
import numpy

def ints_ex_return(df, weight_dict, weight_value):
    '''
    Interceptions without return touchdown
    '''
    ## mask ##
    mask = (
        (df['interception'] == 1) &
        (df['return_touchdown'] != 1)
    )
    ## define true value ##
    val = (
        df['season'].map(weight_dict)
        if weight_dict is not None
        else weight_value
    )
    ## return ##
    return numpy.where(mask, val, 0)
    