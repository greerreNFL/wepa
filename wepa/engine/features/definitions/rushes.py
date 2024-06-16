import pandas as pd
import numpy

def rushes(df, weight_dict, weight_value):
    '''
    All no QB rush attempts
    '''
    ## mask ##
    mask = (
        (df['rush_attempt'] == 1) &
        (df['qb_scramble'] != 1)
    )
    ## define true value ##
    val = (
        df['season'].map(weight_dict)
        if weight_dict is not None
        else weight_value
    )
    ## return ##
    return numpy.where(mask, val, 0)
    