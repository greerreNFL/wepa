import pandas as pd
import numpy

def field_pos_s(df, weight_dict, weight_value):
    '''
    Scaled field position weight with bell curve
    '''
    ## define true value ##
    val = (
        df['season'].map(weight_dict)
        if weight_dict is not None
        else weight_value
    )
    return (
        -val *
        numpy.where(
            (df['yardline_100']/100).fillna(.5) <= .5,
            1 / 
            (
                1+
                numpy.exp(-10*(2*(df['yardline_100']/100).fillna(.5)-0.5))
            )-0.5,
            1 /
            (
                1 +
                numpy.exp(-10*(2*(1-(df['yardline_100']/100).fillna(.5))-0.5))
            ) -
            0.5
        )
    )
    