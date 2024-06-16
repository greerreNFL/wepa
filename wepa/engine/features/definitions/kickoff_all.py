import pandas as pd
import numpy

def kickoff_all(df, weight_dict, weight_value):
    '''
    All kick offs
    '''
    ## mask ##
    mask = (
        (df['play_type']=='kickoff')
    )
    ## define true value ##
    val = (
        df['season'].map(weight_dict)
        if weight_dict is not None
        else weight_value
    )
    ## return ##
    return numpy.where(mask, val, 0)
    