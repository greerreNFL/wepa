import pandas as pd
import numpy

def qb_rush(df, weight_dict, weight_value):
    '''
    All QB rush attempts
    '''
    ## mask ##
    mask = (
        (df['qb_scramble'] == 1) &
        (df['fumble_lost'] != 1)
    )
    ## define true value ##
    val = (
        df['season'].map(weight_dict)
        if weight_dict is not None
        else weight_value
    )
    ## return ##
    return numpy.where(mask, val, 0)
    