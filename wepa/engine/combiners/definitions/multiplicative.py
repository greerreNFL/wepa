import pandas as pd

def multiplicative(df, off_cols, def_cols, params):
    '''
    Combines weights via multiplication
    '''
    ## offensive cols ##
    df['wepa_off'] = df['epa']
    for off_col in off_cols:
        df['wepa_off'] = (
            0 * df[off_col] + 
            df['wepa_off'] * (1-df[off_col])
        )
    ## defensive cols ##
    df['wepa_def'] = df['epa']
    for def_col in def_cols:
        df['wepa_def'] = (
            0 * df[def_col] + 
            df['wepa_def'] * (1-df[def_col])
        )
    ## return ##
    return df

    