import pandas as pd
import numpy

def sacks_w_fumble_ed(df, weight_dict, weight_value):
    '''
    Early down sacks with a lost fumble
    '''
    ## mask ##
    mask = (
        (df['sack'] == 1) &
        (df['fumble_lost'] == 1) &
        (df['down'] <= 2)
    )
    ## define true value ##
    val = (
        df['season'].map(weight_dict)
        if weight_dict is not None
        else weight_value
    )
    ## return ##
    return numpy.where(mask, val, 0)
    