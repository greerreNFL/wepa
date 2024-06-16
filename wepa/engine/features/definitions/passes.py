import pandas as pd
import numpy

def passes(df, weight_dict, weight_value):
    '''
    All pass attempts
    '''
    ## mask ##
    mask = (
        (df['pass_attempt'] == 1)
    )
    ## define true value ##
    val = (
        df['season'].map(weight_dict)
        if weight_dict is not None
        else weight_value
    )
    ## return ##
    return numpy.where(mask, val, 0)
    