import pandas as pd
import numpy

def non_qb_and_tos_half_int(df, weight_dict, weight_value):
    '''
    All turnovers plus non-qb plays, but ints are half discounted
    '''
    ## mask ##
    mask = (
        (
            ## tunrovers ##
            (df['interception'] == 1) |
            (df['fumble_lost'] == 1) |
            ## non-qb ##
            (
                ~(df['pass_attempt'] == 1) &
                ~(df['qb_scramble'] == 1)
            )
        ) &
        ## but not special teams unless fumb ##
        (
            (~numpy.isin(
                df['play_type'],
                ['kickoff','punt','field_goal','extra_point']
            )) &
            ~(df['fumble_lost'] == 1)
        )
    )
    ## define true value ##
    val = (
        df['season'].map(weight_dict)
        if weight_dict is not None
        else weight_value
    )
    ## return ##
    return numpy.where(
        mask,
        numpy.where(
            df['interception'] == 1,
            .5 * val,
            numpy.where(
                df['fumble_lost'] != 1,
                .15 * val,
                val
            )
        ),
        0
    )
    