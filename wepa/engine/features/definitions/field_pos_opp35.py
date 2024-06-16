import pandas as pd
import numpy

def field_pos_opp35(df, weight_dict, weight_value):
    '''
    Inside the opponent 35
    '''
    ## mask ##
    mask = (
        (df['yardline_100'] <= 35)
    )
    ## define true value ##
    val = (
        df['season'].map(weight_dict)
        if weight_dict is not None
        else weight_value
    )
    ## return ##
    return numpy.where(mask, val, 0)
    