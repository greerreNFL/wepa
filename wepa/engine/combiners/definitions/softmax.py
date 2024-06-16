import pandas as pd
import numpy

def softmax(df, off_cols, def_cols, params):
    '''
    Uses softmax to combine weights
    '''
    ## calc the exponents ##
    for col in off_cols + def_cols:
        df['{0}_exp'.format(col)] = numpy.where(
            ## need to set non active col exps to 0 ##
            df[col] == 0,
            0,
            numpy.exp(
                df[col] /
                params['temperature']
            )
        )
    ## calc softmax sums ##
    df['off_sm_sum'] = df[[
        '{0}_exp'.format(i) for i in off_cols
    ]].sum(axis=1)
    df['def_sm_sum'] = df[[
        '{0}_exp'.format(i) for i in def_cols
    ]].sum(axis=1)
    ## since we are using vectorized functions where some rows have a 0 sum,
    ## we need to avoid division by 0 in where functions by setting 
    ## the sum to a value other than 0
    df['off_sm_sum'] = numpy.where(
        df['off_sm_sum'] == 0,
        99, ## arbitrary value
        df['off_sm_sum']
    )
    df['def_sm_sum'] = numpy.where(
        df['def_sm_sum'] == 0,
        99, ## arbitrary value
        df['def_sm_sum']
    )
    ## Calculate the weighted averages ##
    ## Offense ##
    df['wepa_off'] = 0
    for col in off_cols:
        df['wepa_off'] = numpy.where(
            df['off_sm_sum'] == 99,
            ## if no weights are active in the row, wepa is epa
            df['epa'],
            ## else, calc the sm weight and add to the weighted sum
            df['wepa_off'] + (
                ## softmax weight ##
                (
                    df['{0}_exp'.format(col)] /
                    df['off_sm_sum']
                ) *
                ## wepa weight ##
                (
                    0 * df[col] +
                    df['epa'] * (1-df[col])
                )
            )
        )
    ## Defense ##
    df['wepa_def'] = 0
    for col in def_cols:
        df['wepa_def'] = numpy.where(
            df['def_sm_sum'] == 99,
            ## if no weights are active in the row, wepa is epa
            df['epa'],
            ## else, calc the sm weight and add to the weighted sum
            df['wepa_def'] + (
                ## softmax weight ##
                (
                    df['{0}_exp'.format(col)] /
                    df['def_sm_sum']
                ) *
                ## wepa weight ##
                (
                    0 * df[col] +
                    df['epa'] * (1-df[col])
                )
            )
        )
    ## clean up cols ##
    drop_cols = ['off_sm_sum', 'def_sm_sum'] + [
        '{0}_exp'.format(i) for i in off_cols + def_cols
    ]
    df = df.drop(columns=drop_cols)
    ## return df ##
    return df

