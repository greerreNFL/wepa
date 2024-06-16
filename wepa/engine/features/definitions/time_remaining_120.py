import pandas as pd
import numpy

def time_remaining_120(df, weight_dict, weight_value):
    '''
    Inside the two minute warning
    '''
    ## mask ##
    mask = (
        (df['game_seconds_remaining'] < 120)
    )
    ## define true value ##
    val = (
        df['season'].map(weight_dict)
        if weight_dict is not None
        else weight_value
    )
    ## return ##
    return numpy.where(mask, val, 0)
    