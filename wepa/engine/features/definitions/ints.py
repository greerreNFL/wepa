import pandas as pd
import numpy

def ints(df, weight_dict, weight_value):
    '''
    Interceptions
    '''
    ## mask ##
    mask = (
        (df['interception'] == 1)
    )
    ## define true value ##
    val = (
        df['season'].map(weight_dict)
        if weight_dict is not None
        else weight_value
    )
    ## return ##
    return numpy.where(mask, val, 0)
    