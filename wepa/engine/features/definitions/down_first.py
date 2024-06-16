import pandas as pd
import numpy

def down_first(df, weight_dict, weight_value):
    '''
    1st down
    '''
    ## mask ##
    mask = (
        (df['down'] == 1)
    )
    ## define true value ##
    val = (
        df['season'].map(weight_dict)
        if weight_dict is not None
        else weight_value
    )
    ## return ##
    return numpy.where(mask, val, 0)
    