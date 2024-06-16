import pandas as pd
import numpy

def fg_made_s(df, weight_dict, weight_value):
    '''
    Made field goals with a scale based on make prob
    '''
    ## mask ##
    mask = (
        (df['play_type']=='field_goal') &
        (df['field_goal_result'] == 'made')
    )
    ## define true value ##
    val = (
        df['season'].map(weight_dict)
        if weight_dict is not None
        else weight_value
    )
    ## apply s ##
    val = (
        -val *
        numpy.where(
            df['fg_prob'].fillna(.5) <= .5,
            1 / 
            (
                1+
                numpy.exp(-10*(2*df['fg_prob'].fillna(.5)-0.5))
            )-0.5,
            1 /
            (
                1 +
                numpy.exp(-10*(2*(1-df['fg_prob'].fillna(.5))-0.5))
            ) -
            0.5
        )
    )
    ## return ##
    return numpy.where(mask, val, 0)
    