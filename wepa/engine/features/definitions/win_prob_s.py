import pandas as pd
import numpy

def win_prob_s(df, weight_dict, weight_value):
    '''
    Progressive win probability weighting
    '''
    ## define true value ##
    val = (
        df['season'].map(weight_dict)
        if weight_dict is not None
        else weight_value
    )
    ## return along the distribution curve ##
    return (
        -val *
        numpy.where(
            df['wp'].fillna(.5) <= .5,
            1 / 
            (
                1+
                numpy.exp(-10*(2*df['wp'].fillna(.5)-0.5))
            )-0.5,
            1 /
            (
                1 +
                numpy.exp(-10*(2*(1-df['wp'].fillna(.5))-0.5))
            ) -
            0.5
        )
    )

    