import pandas as pd
import numpy

def down_third_short(df, weight_dict, weight_value):
    '''
    3rd down and short (<=3)
    '''
    ## mask ##
    mask = (
        (df['down'] == 3) &
        (df['ydstogo'] <= 3)
    )
    ## define true value ##
    val = (
        df['season'].map(weight_dict)
        if weight_dict is not None
        else weight_value
    )
    ## return ##
    return numpy.where(mask, val, 0)
    