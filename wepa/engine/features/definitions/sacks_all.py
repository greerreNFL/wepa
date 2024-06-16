import pandas as pd
import numpy

def sacks_all(df, weight_dict, weight_value):
    '''
    All sacks
    '''
    ## mask ##
    mask = (
        (df['sack'] == 1)
    )
    ## define true value ##
    val = (
        df['season'].map(weight_dict)
        if weight_dict is not None
        else weight_value
    )
    ## return ##
    return numpy.where(mask, val, 0)
    