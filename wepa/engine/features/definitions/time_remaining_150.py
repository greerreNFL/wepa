import pandas as pd
import numpy

def time_remaining_150(df, weight_dict, weight_value):
    '''
    Time remaining is 2.5 minutes or less
    '''
    ## mask ##
    mask = (
        (df['game_seconds_remaining'] <= 150)
    )
    ## define true value ##
    val = (
        df['season'].map(weight_dict)
        if weight_dict is not None
        else weight_value
    )
    ## return ##
    return numpy.where(mask, val, 0)
    