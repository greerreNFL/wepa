import pandas as pd
import numpy
import pathlib
import importlib
import os

import time

## dictionary for storing feature definitions
features = {}

## get path to feature definitions ##
feature_locations = '{0}/definitions'.format(
    pathlib.Path(__file__).parent.resolve()
)

## build feature dictionary ##
## read each file ##
for file_name in os.listdir(feature_locations):
    ## handle pycache and any other junk in the folder ##
    if not file_name.endswith('.py'):
        continue
    ## import the file as a module ##
    module_name = file_name[:-3]  # Remove the '.py' extension
    module_path = os.path.join(feature_locations, file_name)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ## define function name, which must match file name ##
    ## by remove the extension ##
    feature_name = file_name[:-3]
    ## get feature definition from the module ##
    feature_definition = getattr(
        module, feature_name
    )
    ## add to dictionary ##
    features[feature_name] = feature_definition

class WepaFeature():
    '''
    Feature class that applies weights to DFs
    '''
    def __init__(self, feature_name, weight):
        self.feature_name = feature_name
        self.feature_lookup = None
        self.passed_weight = weight
        self.weight_dict = None
        self.weight_value = None
        self.is_offense = None
        self.feature_definition = None
        self.formatting_exceptions = []
        ## parse inputs ##
        self.parse_feature_name()
        self.parse_weight()
        ## finalize ##
        self.finalize_init()
    
    def parse_feature_name(self):
        '''
        Parses the feature name to determine if it is off or def
        '''
        ## parse the name ##
        if self.feature_name[:2] == 'd_':
            self.feature_lookup = self.feature_name[2:]
            self.is_offense = False
        else:
            self.feature_lookup = self.feature_name
            self.is_offense = True
        ## make sure feature is available  ##
        if self.feature_lookup not in features:
            self.formatting_exceptions.append(
                '     Feature not found in feature definitions!'
            )
    
    def parse_weight(self):
        '''
        Determines if dict or float was passed as weight and assigns
        weight dict and weight value appropriately
        '''
        if isinstance(self.passed_weight, dict):
            self.weight_dict = self.passed_weight
        elif isinstance(self.passed_weight, float) or isinstance(self.passed_weight, int):
            self.weight_value = self.passed_weight
        else:
            self.formatting_exceptions.append(
                '     Weight passed was of type {0}. Must be dict or float'.format(
                    type(self.passed_weight)
                )
            )
    
    def finalize_init(self):
        '''
        If parsing passed, retrieve function
        '''
        ## raise exception if not passed ##
        if len(self.formatting_exceptions) > 0:
            raise Exception('FEATURE ERROR: {0}{1}\n'.format(
                self.feature_name,
                '\n'.join(self.formatting_exceptions)
            ))
        ## pull feature if passed ##
        self.feature_definition = features[self.feature_lookup]

    def update_weight(self, new_passed_weight):
        '''
        Updates the weights of the feature, often passed from
        the optimizer
        '''
        self.passed_weight = new_passed_weight
        ## reset parse ##
        self.weight_dict = None
        self.weight_value = None
        ## reparse ##
        self.parse_weight()

    def add_feature(self, df):
        '''
        Wrapper that applies feature weight to df
        '''
        ## adds weight col to df ##
        df['{0}_weight'.format(
            self.feature_name
        )] = self.feature_definition(
            df=df,
            weight_dict=self.weight_dict,
            weight_value=self.weight_value
        )
        ## return df ##
        return df