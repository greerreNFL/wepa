import pandas as pd
import numpy

def special_teams_all(df, weight_dict, weight_value):
    '''
    All special_teams
    '''
    ## mask ##
    mask = (numpy.isin(
        df['play_type'],
        ['kickoff','punt','field_goal','extra_point']
    ))
    ## define true value ##
    val = (
        df['season'].map(weight_dict)
        if weight_dict is not None
        else weight_value
    )
    ## return ##
    return numpy.where(mask, val, 0)
    