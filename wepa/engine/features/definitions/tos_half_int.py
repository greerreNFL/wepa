import pandas as pd
import numpy

def tos_half_int(df, weight_dict, weight_value):
    '''
    All turnovers, but ints discount by half
    '''
    ## define true value ##
    val = (
        df['season'].map(weight_dict)
        if weight_dict is not None
        else weight_value
    )
    ## return ##
    return numpy.where(
        df['fumble_lost'] == 1,
        val,
        numpy.where(
            df['interception'] == 1,
            .5 * val,
            0
        )
    )
    