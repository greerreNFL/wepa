import pandas as pd
import numpy

def plays_all_boosted(df, weight_dict, weight_value):
    '''
    All plays, without any boolean filtering, but TOs discounted more
    '''
    ## define true value ##
    val = (
        df['season'].map(weight_dict)
        if weight_dict is not None
        else weight_value
    )
    ## return ##
    return numpy.where(
        True,
        numpy.where(
            (df['interception'] == 1) |
            (df['fumble_lost'] == 1),
            val,
            .5 * val
        ),
        0
    )
    