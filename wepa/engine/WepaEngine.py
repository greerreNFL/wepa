import pandas as pd

import time

from .combiners import WepaCombiner
from .features import WepaFeature

class WepaEngine():

    def __init__(self, df, features, weights, combiner, combiner_params=None):
        self.df = df.copy()
        self.features = features
        self.weights = weights
        self.combiner = combiner
        self.wepa_features = []
        self.wepa_combiner = WepaCombiner(combiner, combiner_params)
        self.formatting_exceptions = []
        ## on init ##
        self.finalize_init()

    def check_inputs(self):
        '''
        checks to ensure proper setup for weights and features
        '''
        if not isinstance(self.features, list):
            self.formatting_exceptions.append('     Features must be a list, not {0}'.format(
                type(self.features)
            ))
        if not isinstance(self.weights, list):
            self.formatting_exceptions.append('     Weights must be a list, not {0}'.format(
                type(self.features)
            ))
        if isinstance(self.weights, list) and isinstance(self.features, list):
            if len(self.features) != len(self.weights):
                self.formatting_exceptions.append('     Features and Weights must be of equal length')

    def create_wepa_features(self):
        '''
        Combines features and their weights into a WepaFeatures
        '''
        for index, feature in enumerate(self.features):
            self.wepa_features.append(
                WepaFeature(feature, self.weights[index])
            )    

    def finalize_init(self):
        '''
        Raise any formatting errors that arose. Create features if passed
        '''
        if len(self.formatting_exceptions) > 0:
            raise Exception('ENGINE ERROR:{1}'.format(
                '\n'.join(self.formatting_exceptions)
            ))
        ## pull feature if passed ##
        self.create_wepa_features()
    
    def update_weights(self, weights):
        '''
        Updates the engine weights without re-copying the PBP
        '''
        ## update weights ##
        if len(weights) == len(self.wepa_features):
            self.weights = weights
        else:
            raise Exception('ENGINE ERROR: Update weight length did not match features:\n    Features:{0}\n     Weights:{1}'.format(
                self.features,
                weights
            ))
        ## update weights ##
        for index, feature in enumerate(self.wepa_features):
            feature.update_weight(weights[index])
    
    def update_combiner(self, combiner=None, combiner_params=None):
        '''
        Reinits the combiner
        '''
        combiner_ = combiner if combiner is not None else self.combiner
        combiner_params_ = combiner_params if combiner_params is not None else self.combiner_params
        self.combiner = combiner_
        self.wepa_combiner = WepaCombiner(combiner_, combiner_params_)

    def apply_wepa(self):
        '''
        Applies specified feautures to a df
        '''
        for feature in self.wepa_features:
            ## apply the weight
            self.df = feature.add_feature(self.df)
        ## combine individual weights
        self.df = self.wepa_combiner.combine_weights(self.df, self.features)