import pandas as pd
import numpy

from wepa.wepa.engine.features.feature_utils import tail_scale

def time_remaining_ts(df, weight_dict, weight_value):
    '''
    Use a tail scale to progressive discount the end of game
    '''
    ## create a progressive deweighting of the 4th quarter ##
    ## since tail scale is built for probs, everything else should
    ## be "50%"
    df['seconds_scaled'] = numpy.where(
        df['game_seconds_remaining'] <= 900,
        .5 * (df['game_seconds_remaining']/900),
        .5
    )
    ## define true value ##
    val = (
        df['season'].map(weight_dict)
        if weight_dict is not None
        else weight_value
    )
    ## return ##
    return tail_scale(df, 'seconds_scaled', val)
    