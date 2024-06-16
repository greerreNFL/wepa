import pandas as pd
import numpy

def win_prob_lt_10(df, weight_dict, weight_value):
    '''
    Garbage time (10% WP)
    '''
    ## mask ##
    prob_thresh = .1
    mask = (
        (df['wp'] >= 1-prob_thresh) |
        (df['wp'] <=   prob_thresh)
    )
    ## define true value ##
    val = (
        df['season'].map(weight_dict)
        if weight_dict is not None
        else weight_value
    )
    ## return ##
    return numpy.where(mask, val, 0)
    