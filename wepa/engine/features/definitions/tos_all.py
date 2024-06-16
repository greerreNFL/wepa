import pandas as pd
import numpy

def tos_all(df, weight_dict, weight_value):
    '''
    All turnovers
    '''
    ## mask ##
    mask = (
        (
            ## tunrovers ##
            (df['interception'] == 1) |
            (df['fumble_lost'] == 1)
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
    