import pandas as pd
import numpy

from wepa.wepa.engine.features.feature_utils import tail_scale

def field_pos_ts(df, weight_dict, weight_value):
    '''
    Scaled field position with a tail scale
    '''
    ## translate yardage into a pct for the tail scale function ##
    df['yardline_scaled'] = df['yardline_100']/100
    ## define true value ##
    val = (
        df['season'].map(weight_dict)
        if weight_dict is not None
        else weight_value
    )
    ## return ##
    return tail_scale(df, 'yardline_scaled', val)
    