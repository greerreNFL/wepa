import pandas as pd
import numpy

from wepa.wepa.engine.features.feature_utils import tail_scale

def win_prob_ts_asym(df, weight_dict, weight_value):
    '''
    Progressive win probability weighting using tail scale, but
    with an asymetric distribution, where an offense that is behind
    has a higher chance to be deweighted
    '''
    ## define true value ##
    df['wp_asym_scale'] = numpy.where(
        df['wp'].fillna(.5) >= .5,
        0.5 * 0.20 + 0.80 * df['wp'].fillna(.5),
        0.0 * 0.20 + 0.80 * df['wp'].fillna(.5)
    )
    val = (
        df['season'].map(weight_dict)
        if weight_dict is not None
        else weight_value
    )
    ## return along the distribution curve ##
    return tail_scale(df, 'wp_asym_scale', val)

    