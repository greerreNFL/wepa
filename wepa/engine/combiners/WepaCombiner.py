import pandas as pd
import numpy
import pathlib
import importlib
import os

## dictionary for storing combiner definitions
combiners = {}

## get path to feature definitions ##
combiner_locations = '{0}/definitions'.format(
    pathlib.Path(__file__).parent.resolve()
)

## build combiner dictionary ##
## read each file ##
for file_name in os.listdir(combiner_locations):
    ## handle pycache and any other junk in the folder ##
    if not file_name.endswith('.py'):
        continue
    ## import the file as a module ##
    module_name = file_name[:-3]  # Remove the '.py' extension
    module_path = os.path.join(combiner_locations, file_name)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ## define function name, which must match file name ##
    ## by remove the extension ##
    combiner_name = file_name[:-3]
    ## get combiner definition from the module ##
    combiner_definition = getattr(
        module, combiner_name
    )
    ## add to dictionary ##
    combiners[combiner_name] = combiner_definition

class WepaCombiner():
    '''
    Combiner class that combines weights into off and def wepas
    '''
    def __init__(self, combiner_name, combiner_params=None):
        self.combiner_name = combiner_name
        self.combiner_params = combiner_params
        self.combiner_definition = None
        self.formatting_exceptions = []
        ## parse inputs ##
        self.parse_combiner_name()
        ## finalize ##
        self.finalize_init()
    
    def parse_combiner_name(self):
        '''
        Parses the combiner name to determine if it is off or def
        '''
        ## make sure feature is available  ##
        if self.combiner_name not in combiners:
            self.formatting_exceptions.append(
                '     Combiner not found in combiner definitions!'
            )
    
    def finalize_init(self):
        '''
        If parsing passed, retrieve combiner
        '''
        ## raise exception if not passed ##
        if len(self.formatting_exceptions) > 0:
            raise Exception('COMBINER ERROR: {0}{1}'.format(
                self.combiner_name,
                '\n'.join(self.formatting_exceptions)
            ))
        ## pull feature if passed ##
        self.combiner_definition = combiners[self.combiner_name]

    def split_cols(self, cols):
        '''
        Splits a unified set of wepa cols into off and def
        '''
        ## struct ##
        off_cols = []
        def_cols = []
        for col in cols:
            if col[:2] == 'd_':
                def_cols.append('{0}_weight'.format(col))
            else:
                off_cols.append('{0}_weight'.format(col))
        ## return ##
        return off_cols, def_cols

    def combine_weights(self, df, features):
        '''
        Wrapper that applies calcs off and def wepa for
        a df
        '''
        ## get off and def cols ##
        off_cols, def_cols = self.split_cols(features)
        ## combine ##
        df = self.combiner_definition(
            df,
            off_cols,
            def_cols,
            self.combiner_params
        )
        ## return df ##
        return df