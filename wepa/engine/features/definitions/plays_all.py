import pandas as pd
import numpy

def plays_all(df, weight_dict, weight_value):
    '''
    All plays, without any boolean filtering
    '''
    ## define true value ##
    val = (
        df['season'].map(weight_dict)
        if weight_dict is not None
        else weight_value
    )
    ## return ##
    return numpy.where(True, val, 0)
    