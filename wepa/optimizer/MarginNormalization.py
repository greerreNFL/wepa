import pandas as pd
import numpy
import pathlib
import json

import statsmodels.api as sm

class MarginNormalization:
    '''
    A trainer that creates a model for normalizing the wepa
    margin against the actual MoV to make a less predictive, 
    but more apples to apples comparison
    '''
    def __init__(self):
        self.package_root = pathlib.Path(__file__).parent.parent.parent.resolve()
        self.config_loc = '{0}/model_config.json'.format(self.package_root)
        self.m = 0
        self.b = 0

    def train_normalizer(self):
        '''
        Trains the normalization model
        '''
        ## load wepa games ##
        df = pd.read_csv(
            '{0}/data/wepa_by_game_at.csv'.format(self.package_root),
            index_col=0
        )
        ## create an intercept ##
        df['constant'] = 1
        ## create a model ##
        model = sm.OLS(
            df['margin'],
            df[['wepa_net', 'constant']]
        ).fit()
        ## set values ##
        self.m = model.params[0]
        self.b = model.params[1]

    
    def update_config(self):
        '''
        Runs the normalization and then writes the results to the config
        '''
        ## train ##
        self.train_normalizer()
        ## open config ##
        with open(self.config_loc, 'r') as file:
            config = json.load(file)
        ## update values ##
        config['normalization']['m'] = self.m
        config['normalization']['b'] = self.b
        ## save ##
        with open(self.config_loc, 'w') as file:
            json.dump(config, file, indent=4)
