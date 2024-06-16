import pandas as pd
import numpy

def fumble_w_return_td(df, weight_dict, weight_value):
    '''
    All lost fumbles that were returned for TDs
    '''
    ## mask ##
    mask = (
        (df['fumble_lost'] == 1) &
        (df['return_touchdown'] == 1)
    )
    ## define true value ##
    val = (
        df['season'].map(weight_dict)
        if weight_dict is not None
        else weight_value
    )
    ## return ##
    return numpy.where(mask, val, 0)
    