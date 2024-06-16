import pandas as pd
import numpy

def fumble_rush(df, weight_dict, weight_value):
    '''
    All lost fumbles on rush attempts
    '''
    ## mask ##
    mask = (
        (df['fumble_lost'] == 1) &
        (df['rush_attempt'] == 1)
    )
    ## define true value ##
    val = (
        df['season'].map(weight_dict)
        if weight_dict is not None
        else weight_value
    )
    ## return ##
    return numpy.where(mask, val, 0)
    