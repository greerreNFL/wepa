import pandas as pd
import numpy

def down_fourth(df, weight_dict, weight_value):
    '''
    4th down (ex fg, punt, etc)
    '''
    ## mask ##
    mask = (
        (df['down'] == 4) &
        (
            (df['pass_attempt'] == 1) |
            (df['rush_attempt'] == 1)
        )
    )
    ## define true value ##
    val = (
        df['season'].map(weight_dict)
        if weight_dict is not None
        else weight_value
    )
    ## return ##
    return numpy.where(mask, val, 0)
    