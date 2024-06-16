import pandas as pd
import numpy

from wepa.wepa.engine.features.feature_utils import tail_scale

def fg_made_ts(df, weight_dict, weight_value):
    '''
    Made field goals with a scale based on make prob using tail scale
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
    ts_vals = tail_scale(df, 'fg_prob', val)
    ## return ##
    return numpy.where(mask, ts_vals, 0)
    