import pandas as pd
import numpy
import math

import wepa.wepa.optimizer as optimizer
## will need to load the wepa optimizer ##

class WepaWeights():
    '''
    Handler for loading weights and checking if they are up to date
    This class will update the weights if they are missing or not up
    to date.

    Once weights are up to date, the get_weights function can be used to
    return the optimal weight for all time, or by point in time seasons
    '''
    def __init__(self, data, config):
        self.data = data
        self.config = config
        self.passed_features = config['features']
        self.weights_point_in_time, self.weights = self.load_weights()
        self.required_update = self.check_for_update()
        ## handle update on load ##
        self.handle_update()

    def load_point_in_time(self):
        '''
        Loads the point in time weights
        '''
        try:
            return pd.read_csv(
                '{0}/weights_point_in_time.csv'.format(
                    self.data.data_folder
                ),
                index_col=0
            )
        except:
            return None
    
    def load_all_time_weights(self):
        '''
        Loads the all time weights
        '''
        try:
            return pd.read_csv(
                '{0}/weights.csv'.format(
                    self.data.data_folder
                ),
                index_col=0
            )
        except:
            return None

    def load_weights(self):
        '''
        wrapper for weight loading
        '''
        return self.load_point_in_time(), self.load_all_time_weights()
    
    def check_for_update(self):
        '''
        Checks the loaded weights and determines if an update is required
        '''
        ## if either weights failed to load, update ##
        if self.weights is None or self.weights_point_in_time is None:
            print('     Weights were not found. Will retrain...')
            return True
        ## check that features in saved weights match the passed features ##
        ## populate a list of features with trained weights
        features = []
        ## weights ##
        for col in self.weights.columns.to_list():
            if col not in [
                'trained_through', 'season', 'model_version', 'model_rsq',
                'epa_rsq', 'margin_rsq', 'rsq_lift', 'features', 'combiner'
            ]:
                features.append(col)
        ## point in time ##
        for col in self.weights_point_in_time.columns.to_list():
            if col not in [
                'trained_through', 'season', 'model_version', 'model_rsq',
                'epa_rsq', 'margin_rsq', 'rsq_lift', 'features', 'combiner',
                 ## for point in time, we also measure the forward rsq
                'forward_model_rsq_next_season', 'forward_margin_rsq_next_season',
                'forward_rsq_lift_next_season',
                'forward_model_rsq_all_future_seasons',
                'forward_margin_rsq_all_future_seasons',
                'forward_rsq_lift_all_future_seasons'
            ]:
                features.append(col)
        ## make check ##
        for feature in features:
            if feature not in self.passed_features:
                print('     Existing weights trained on features that are now not included. Will retrain...')
                print('          {0}'.format(feature))
                return True
        for feature in self.passed_features:
            if feature not in features:
                print('     New features passed. Will retrain...')
                print('          {0}'.format(feature))
                return True
        ## check that model is trained through most recent season ##
        ## get trained through from weights ##
        trained_through = min(
            self.weights['trained_through'].max(),
            self.weights_point_in_time['trained_through'].max()
        )
        if trained_through < self.data.last_completed_season:
            print('     Weights are trained through {0}, while last completed season is {1}. Will retrain...'.format(
                trained_through, self.data.last_completed_season
            ))
            return True

    def update_weights(self):
        '''
        Updates Weights
        '''
        ## create a wepa optimizer ##
        trainer = optimizer.WepaOptimizer(
            train_df=self.data.pbp[
                self.data.pbp['season'] <= self.data.last_completed_season
            ].copy(),
            features=self.config['features'],
            combiner=self.config['combiner'],
            combiner_params=self.config['combiner_params'],
            window_type='team_halves'
        )
        print('     Updating wepa weights...')
        ## train ##
        trainer.optimize()
        ## structure output ##
        rec = {
            'model_version' : self.config['version'],
            'trained_through' : trainer.train_df['season'].max(),
            'model_rsq' : trainer.opti_rec['wepa_rsq'],
            'epa_rsq' : trainer.opti_rec['epa_rsq'],
            'margin_rsq' : trainer.opti_rec['margin_rsq'],
            'rsq_lift' : trainer.opti_rec['wepa_rsq'] / trainer.opti_rec['epa_rsq'] - 1,
            'features' : trainer.opti_rec['features'],
            'combiner' : self.config['combiner']
        }
        ## add weights ##
        for index, feature in enumerate(trainer.features):
            rec[feature] = trainer.opti_weights[index]
        ## store ##
        self.weights = pd.DataFrame([rec])
        ## save ##
        self.weights.to_csv(
            '{0}/weights.csv'.format(
                self.data.data_folder
            )
        )

    
    def smoother(self, trainer, median_array, value_array, seasons):
        '''
        Helper function for applying smoothing
        '''
        ## update medians ##
        if len(median_array) == 0:
            ## init if necessary #
            median_array = [[x] for x in trainer.opti_weights]
        else:
            for i,v in enumerate(trainer.opti_weights):
                median_array[i].append(trainer.opti_weights[i])
        ## increment season counter ##
        seasons += 1
        ## create softmax weights w/ progressive median ##
        temp = .25
        ## apply exp ##
        med_exp = math.exp((
            self.config['smoothing']['median'] +
            (seasons / 15)
        ) / temp)
        prev_exp = math.exp(self.config['smoothing']['ema'] / temp)
        new_exp = math.exp((
            1 -
            self.config['smoothing']['median'] -
            self.config['smoothing']['ema']
        ) / temp)
        ## sm ##
        med_sm = med_exp / (med_exp + prev_exp + new_exp)
        prev_sm = prev_exp / (med_exp + prev_exp + new_exp)
        new_sm = new_exp / (med_exp + prev_exp + new_exp)
        ## update values ##
        if seasons >= self.config['smoothing']['start']:
            ## init if necessary ##
            if len(value_array) == 0:
                value_array = trainer.opti_weights
            ## apply smooth ##
            for i,v in enumerate(trainer.opti_weights):
                ## get median ##
                med = numpy.median(median_array[i])
                ## apply smooth ##
                trainer.opti_weights[i] = (
                    med_sm * med +
                    prev_sm * value_array[i] +
                    new_sm * v
                )
                ## update smoothed vals ##
                value_array[i] = trainer.opti_weights[i]
        ## return ##
        return trainer, median_array, value_array, seasons

    def update_weights_point_in_time(self):
        '''
        Updates the point in time weights
        '''
        print('     Updating pointing in time wepa weights')
        ## container for all records ##
        recs = []
        ## containers smoothing ##
        median_array = []
        value_array = [] 
        seasons = 0
        for season in range(
            self.data.first_completed_season, self.data.last_completed_season+1
        ):
            print('          On {0}...'.format(
                season
            ))            
            ## create a wepa optimizer ##
            trainer = optimizer.WepaOptimizer(
                train_df=self.data.pbp[
                    self.data.pbp['season'] <= season
                ].copy(),
                features=self.config['features'],
                combiner=self.config['combiner'],
                combiner_params=self.config['combiner_params'],
                window_type='team_halves'
            )
            ## train ##
            trainer.optimize()
            ## apply smoothing ##
            trainer, median_array, value_array, seasons = self.smoother(
                trainer, median_array, value_array, seasons
            )
            ## test next season ##
            tester_ns = optimizer.WepaOptimizer(
                train_df=self.data.pbp[
                    self.data.pbp['season'] == season + 1
                ].copy(),
                features=self.config['features'],
                combiner=self.config['combiner'],
                combiner_params=self.config['combiner_params'],
                window_type='team_halves'
            )
            test_rsqs_ns = tester_ns.calc_rsqs(trainer.opti_weights) if season < self.data.last_completed_season else {
                'wepa' : numpy.nan,
                'margin' : numpy.nan,
                'epa' : numpy.nan
            } 
            ## test all future ##
            tester_fs = optimizer.WepaOptimizer(
                train_df=self.data.pbp[
                    self.data.pbp['season'] > season
                ].copy(),
                features=self.config['features'],
                combiner=self.config['combiner'],
                combiner_params=self.config['combiner_params'],
                window_type='team_halves'
            )
            test_rsqs_fs = tester_fs.calc_rsqs(trainer.opti_weights) if season < self.data.last_completed_season else {
                'wepa' : numpy.nan,
                'margin' : numpy.nan,
                'epa' : numpy.nan
            }
            ## structure output ##
            rec = {
                'model_version' : self.config['version'],
                'trained_through' : trainer.train_df['season'].max(),
                'model_rsq' : trainer.opti_rec['wepa_rsq'],
                'epa_rsq' : trainer.opti_rec['epa_rsq'],
                'margin_rsq' : trainer.opti_rec['margin_rsq'],
                'rsq_lift' : trainer.opti_rec['lift'],
                'features' : trainer.opti_rec['features'],
                'combiner' : self.config['combiner'],
                'forward_model_rsq_next_season' : test_rsqs_ns['wepa'],
                'forward_margin_rsq_next_season' : test_rsqs_ns['epa'],
                'forward_rsq_lift_next_season' : test_rsqs_ns['wepa'] / test_rsqs_ns['epa'] - 1,
                'forward_model_rsq_all_future_seasons' : test_rsqs_fs['wepa'],
                'forward_margin_rsq_all_future_seasons' : test_rsqs_fs['epa'],
                'forward_rsq_lift_all_future_seasons' : test_rsqs_fs['wepa'] / test_rsqs_fs['epa'] - 1
            }
            ## add weights ##
            for index, feature in enumerate(trainer.features):
                rec[feature] = trainer.opti_weights[index]
            ## add to record container ##
            recs.append(rec)
        ## save ##
        pd.DataFrame(recs).to_csv(
            '{0}/weights_point_in_time.csv'.format(
                self.data.data_folder
            )
        )

    def handle_update(self):
        '''
        If the weights need to be updated, it will handle the update
        '''
        if self.required_update:
            self.update_weights()
            self.update_weights_point_in_time()
    
    def get_weights(self):
        '''
        Returns weights in a format for the optimizer
        '''
        ## ensure again that the weights exist ##
        if self.weights_point_in_time is None or self.weights is None:
            print('     Attempted to load weights that do not exist. Will update')
            self.required_update = True
            self.handle_update()
        ## all time weights ##
        at_weights = []
        for feature in self.passed_features:
            at_weights.append(self.weights.iloc[0][feature])
        ## point in time ##
        pit_weights = [{} for n in range(0, len(self.passed_features))] ## init dicts
        for index, row in self.weights_point_in_time.iterrows():
            for feature_index, feature in enumerate(self.passed_features):
                ## add a season so when the point in time weight is applied, it
                ## is only using historic data
                pit_weights[feature_index][row['trained_through']+1] = row[feature]
        ## return ##
        return at_weights, pit_weights

        