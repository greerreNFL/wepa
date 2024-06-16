import pandas as pd
import numpy

def mean(df, off_cols, def_cols, params):
    '''
    Combines weights via a simple average
    '''
    ## offensive cols ##
    ## two cols to keep enable vectorized conditional avg ##
    df['active_count'] = 0
    df['active_sum'] = 0
    for col in off_cols:
        ## add to sum if active ##
        df['active_sum'] = df['active_sum'] + numpy.where(
            df[col] != 0,
            (
                0 * df[col] +
                df['epa'] * (1 - df[col])
            ),
            0
        )
        ## add count if active ##
        df['active_count'] = df['active_count'] + numpy.where(
            df[col] != 0,
            1,
            0
        )
    ## create the conditional avg ##
    ## handle instance where no weight is active and avoid div by 0 ##
    df['active_sum'] = numpy.where(
        df['active_count'] == 0,
        df['epa'],
        df['active_sum']
    )
    df['active_count'] = numpy.where(
        df['active_count'] == 0,
        1,
        df['active_count']
    )
    df['wepa_off'] = df['active_sum'] / df['active_count']
    ## def cols ##
    ## two cols to keep enable vectorized conditional avg ##
    df['active_count'] = 0
    df['active_sum'] = 0
    for col in def_cols:
        ## add to sum if active ##
        df['active_sum'] = df['active_sum'] + numpy.where(
            df[col] != 0,
            (
                0 * df[col] +
                df['epa'] * (1 - df[col])
            ),
            0
        )
        ## add count if active ##
        df['active_count'] = df['active_count'] + numpy.where(
            df[col] != 0,
            1,
            0
        )
    ## create the conditional avg ##
    ## handle instance where no weight is active and avoid div by 0 ##
    df['active_sum'] = numpy.where(
        df['active_count'] == 0,
        df['epa'],
        df['active_sum']
    )
    df['active_count'] = numpy.where(
        df['active_count'] == 0,
        1,
        df['active_count']
    )
    df['wepa_def'] = df['active_sum'] / df['active_count']
    df = df.drop(columns=['active_sum', 'active_count']).copy()
    ## return ##
    return df

    