import pandas as pd
import numpy

def fg_miss(df, weight_dict, weight_value):
    '''
    Missed field goals ex blocks
    '''
    ## mask ##
    mask = (
        (df['play_type']=='field_goal') &
        (df['field_goal_result'] == 'missed')
    )
    ## define true value ##
    val = (
        df['season'].map(weight_dict)
        if weight_dict is not None
        else weight_value
    )
    ## return ##
    return numpy.where(mask, val, 0)
    