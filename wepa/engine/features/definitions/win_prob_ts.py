import pandas as pd
import numpy

from wepa.wepa.engine.features.feature_utils import tail_scale

def win_prob_ts(df, weight_dict, weight_value):
    '''
    Progressive win probability weighting using tail scale
    '''
    ## define true value ##
    val = (
        df['season'].map(weight_dict)
        if weight_dict is not None
        else weight_value * numpy.ones(len(df))
    )
    ## return along the distribution curve ##
    return tail_scale(df, 'wp', val)

    