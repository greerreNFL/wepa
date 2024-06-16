import pandas as pd
import numpy
import statsmodels.api as sm
from scipy.optimize import minimize, basinhopping
import time
import wepa.wepa.engine as engine
from .windows import WepaWindow

class WepaOptimizer():
    '''
    Takes a training df and optimizes it for oos rsq
    '''
    def __init__(self,
            train_df, features, combiner, combiner_params,
            best_guesses=None, bound=(0,1), window_type='rand',
            scale=3, tol=0.000001, step=0.00001, method='SLSQP',
            basin_hop=False
        ):
        self.train_df = train_df
        self.window_type = window_type
        self.window_df = self.generate_windows(train_df, window_type)
        self.features = features
        self.combiner = combiner
        self.combiner_params = combiner_params
        ## opti params ##
        self.best_guesses = best_guesses
        self.bound = bound
        self.bounds = tuple((0, 1) for _ in range(len(features))) ## normalized
        self.scale = scale
        self.tol = tol
        self.step = step
        self.method = method
        self.basin_hop = basin_hop
        ## wepa engine ##
        self.wepa_engine = None
        ## post optimization vars ##
        self.opti_weights = []
        self.opti_rsq = 0
        self.opti_rsq_baseline = 0
        self.opti_seconds = 0
        self.opti_rec = {}
        ## init ##
        self.init_engine()

    def generate_windows(self, pbp, window_type, obs_min=3):
        '''
        Splits data set into in sample and oos games
        '''
        ## init a WepaWindow ##
        windower = WepaWindow(pbp)
        ## get requested window type with a check for min obs ##
        passing = False
        times_run = 0
        windowed = None
        while not passing and times_run < 6:
            windowed = windower.gen_window(window_type)
            ## check if there are games across both ##
            ## isolate in sample ##
            is_agg = windowed[
                windowed['window'] == 'in_sample'
            ].groupby(['team', 'season']).agg(
                is_obs = ('season', 'count')
            ).reset_index()
            ## isolate out of sample
            oos_agg = windowed[
                windowed['window'] == 'out_of_sample'
            ].groupby(['team', 'season']).agg(
                oos_obs = ('season', 'count')
            ).reset_index()
            ## merge ##
            agg = pd.merge(
                is_agg,
                oos_agg,
                on=['team', 'season'],
                how='outer'
            )
            agg['oos_obs'] = agg['oos_obs'].fillna(0)
            agg['is_obs'] = agg['is_obs'].fillna(0)
            ## check that sampling met threshold ##
            if (
                agg['oos_obs'].min() >= obs_min and
                agg['is_obs'].min() >= obs_min
            ):
                passing = True
            else:
                times_run += 1
        ## set window ##
        if not passing:
            print('          Warning -- sampling is below observation threshold')
        return windowed
    
    def init_engine(self):
        '''
        Initialize a WepaEngine to use in the opti
        '''
        self.wepa_engine = engine.WepaEngine(
            df=self.train_df,
            features=self.features,
            weights=[0]*len(self.features),
            combiner=self.combiner,
            combiner_params=self.combiner_params
        )
    
    def update_features(self, new_features):
        '''
        Updates the model features and re-inits. This can be useful if you want to
        train multiple models over the same windows
        '''
        self.features=new_features
        self.bounds=tuple((0, 1) for _ in range(len(new_features))) ## normalized
        self.opti_weights = []
        self.opti_rsq = 0
        self.opti_rsq_baseline = 0
        self.opti_seconds = 0
        self.opti_rec = {}
        self.init_engine()
    
    def update_combiner(self, combiner=None, combiner_params=None):
        '''
        Updates the combiner and its params
        '''
        self.wepa_engine.update_combiner(
            combiner=combiner,
            combiner_params=combiner_params
        )

    def rewindow(self, window_type='rand'):
        '''
        Rewindows the dataset
        '''
        ## update window type ##
        self.window_type=window_type
        self.window_df = self.generate_windows(
            self.train_df, window_type
        )
    
    def calc_wepa(self, x):
        '''
        Calculates wepa for a given set of weights from the opti
        '''
        ## update weights ##
        self.wepa_engine.update_weights(x)
        self.wepa_engine.apply_wepa()

    def normalize_weights(self, x):
        '''
        Takes weights and normalizes them to 0-1 for the optimizer
        '''
        normalized_weights = [
            (weight-self.bound[0]) / (self.bound[1] - self.bound[0])
            for weight in x
        ]
        return normalized_weights

    def denormalize_weights(self, x_norm):
        '''
        Returns normalized 0-1 weights back to their original vals
        '''
        denormalized_weights = [
            (nw * (self.bound[1] - self.bound[0]) + self.bound[0])
            for nw in x_norm
        ]
        return denormalized_weights
    
    def generate_normalized_guesses(self):
        '''
        Generates normalized weights from the best guesses
        '''
        ## init best guesses if none passed ##
        ## set these as the midpoint between the bounds
        if self.best_guesses is None:
            self.best_guesses = []
            for i in self.bounds:
                self.best_guesses.append((i[1]+i[0])/2)
        ## normalize ##
        return self.normalize_weights(self.best_guesses)
    
    def agg_games(self):
        '''
        Aggregates wepa at the game level
        '''
        ## off agg ##
        off_agg = self.wepa_engine.df.groupby(['posteam', 'game_id', 'season']).agg(
            wepa_off = ('wepa_off', 'sum'),
            epa_off = ('epa', 'sum'),
        ).reset_index().rename(columns={
            'posteam' : 'team'
        })
        ## def agg ##
        def_agg = self.wepa_engine.df.groupby(['defteam', 'game_id', 'season']).agg(
            wepa_def = ('wepa_def', 'sum'),
            epa_def = ('epa', 'sum'),
        ).reset_index().rename(columns={
            'defteam' : 'team'
        })
        ## join ##
        agg = pd.merge(
            off_agg,
            def_agg,
            on=['game_id', 'team', 'season'],
            how='left'
        )
        agg['wepa_margin'] = agg['wepa_off'] - agg['wepa_def']
        agg['epa_margin'] = agg['epa_off'] - agg['epa_def']
        ## return ##
        return agg

    def split_agg_by_window(self, agg_df):
        '''
        Splits data into in sample vs out of sample and pivots along sample
        '''
        ## combine games to window ##
        rsq_df = pd.merge(
            self.window_df,
            agg_df,
            on=['game_id', 'team', 'season'],
            how='left'
        )
        ## add a plug for actual margin to account for things not embedded into EPA ##
        ## This effectively captures the inherent value of possessions
        rsq_df['margin_plug'] = rsq_df['margin'] - rsq_df['epa_margin']
        rsq_df['epa_adj_margin'] = rsq_df['epa_margin'] + rsq_df['margin_plug']
        rsq_df['wepa_margin'] = rsq_df['wepa_margin'] + rsq_df['margin_plug']
        ## group by window ##
        rsq_df = rsq_df.groupby([
            'team', 'season', 'window'
        ]).agg(
            obs = ('margin', 'count'),
            avg_margin = ('margin', 'mean'),
            avg_wepa_margin = ('wepa_margin', 'mean'),
            avg_epa_margin = ('epa_margin', 'mean'),
        ).reset_index()
        ## de flatten ##
        rsq_df = pd.merge(
            ## in sample ##
            rsq_df[
                rsq_df['window'] == 'in_sample'
            ][[
                'team', 'season', 'obs', 'avg_margin',
                'avg_wepa_margin', 'avg_epa_margin'
            ]].rename(columns={
                'obs' : 'obs_is',
                'avg_margin' : 'avg_margin_is',
                'avg_wepa_margin' : 'avg_wepa_is',
                'avg_epa_margin' : 'avg_epa_is',
            }),
            ## out of sample
            rsq_df[
                rsq_df['window'] == 'out_of_sample'
            ][[
                'team', 'season', 'obs', 'avg_margin',
                'avg_wepa_margin', 'avg_epa_margin'
            ]].rename(columns={
                'obs' : 'obs_oos',
                'avg_margin' : 'avg_margin_oos',
                'avg_wepa_margin' : 'avg_wepa_oos',
                'avg_epa_margin' : 'avg_epa_oos',
            }),
            on=['team', 'season'],
            how='left'
        )
        ## return ##
        return rsq_df
    
    def calc_rsqs(self, x, obj='margin'):
        '''
        Wrapper func that applies wepa and calculates the rsq
        '''
        ## apply wepa using weights passed ##
        self.calc_wepa(x)
        ## create game level aggregations ##
        agg = self.agg_games()
        ## split into in and out of sample ##
        rsq_df = self.split_agg_by_window(agg)
        ## calculate rsqs ##
        rsqs = {}
        for measure in ['margin', 'wepa', 'epa']:
            ## for each measure, create a model to oos margin ##
            ## drop nans which will throw error ##
            rsq_df_temp = rsq_df[
                (~pd.isnull(rsq_df['avg_{0}_oos'.format(obj)])) &
                (~pd.isnull(rsq_df['avg_{0}_is'.format(measure)]))
            ].copy()
            if len(rsq_df_temp) < len(rsq_df):
                print('               Warning -- the rsq dataframe had NaNs from random sampling')
                t = rsq_df[
                    (pd.isnull(rsq_df['avg_{0}_oos'.format(obj)])) |
                    (pd.isnull(rsq_df['avg_{0}_is'.format(measure)]))
                ]
                print(t)
            ## run regression
            model = sm.OLS(
                rsq_df_temp['avg_{0}_oos'.format(obj)],
                rsq_df_temp['avg_{0}_is'.format(measure)]
            ).fit()
            ## add rsq to rsqs ##
            rsqs[measure] = model.rsquared
        ## return ##
        return rsqs

    def obj_func(self, x, obj):
        '''
        Objective function that the optimizer
        '''
        ## denormalize weights ##
        x_denorm = self.denormalize_weights(x)
        ## first get rsqs ##
        rsqs = self.calc_rsqs(x_denorm, obj)
        return ((1 - rsqs['wepa']) ** self.scale) * (self.scale ** 2)

    def optimize(self, obj='margin'):
        '''
        Function that performs the optimization
        '''
        ## optimize ##
        ## reset counter ##
        self.opti_round = 0
        opti_time_start = float(time.time())
        if self.basin_hop:
            solution = basinhopping(
                self.obj_func,
                self.generate_normalized_guesses(),
                minimizer_kwargs={
                    'method' : self.method,
                    'args' : (obj),
                    'bounds' : self.bounds,
                    'options' :{
                        'ftol' : self.tol,
                        'eps' : self.step
                    }
                }
            )
        else:
            solution = minimize(
                self.obj_func,
                self.generate_normalized_guesses(),
                args=(obj),
                bounds=self.bounds,
                method=self.method,
                options={
                    'ftol' : self.tol,
                    'eps' : self.step
                }
            )
        if not solution.success:
            print('     FAIL')
        opti_time_end = float(time.time())
        ## update properties ##
        self.opti_seconds = opti_time_end - opti_time_start
        self.opti_weights = self.denormalize_weights(solution.x)
        ## generate rsqs ##
        rsqs = self.calc_rsqs(
            x=self.opti_weights,
            obj=obj
        )
        self.opti_rsq = rsqs['wepa']
        self.opti_rsq_baseline  = rsqs['epa']
        ## construct the record ##
        self.opti_rec = {}
        self.opti_rec['run_time'] = self.opti_seconds
        self.opti_rec['iterations'] = solution.nit
        self.opti_rec['avg_time_per_eval'] = self.opti_seconds / solution.nit
        try:
            self.opti_rec['jacobian'] = solution.jac
        except:
            ## if using basin hop, this will not be avail ##
            self.opti_rec['jacobian'] = numpy.nan
        self.opti_rec['wepa_rsq'] = self.opti_rsq
        self.opti_rec['epa_rsq'] = self.opti_rsq_baseline
        self.opti_rec['margin_rsq'] = rsqs['margin']
        self.opti_rec['lift'] = self.opti_rsq - self.opti_rsq_baseline
        self.opti_rec['lift_over_margin'] = self.opti_rsq - rsqs['margin']
        self.opti_rec['features'] = ', '.join(self.features)
        for index, feature in enumerate(self.features):
            self.opti_rec[feature] = self.opti_weights[index]
            self.opti_rec['{0}_plays_pct'.format(feature)] = (
                len(self.wepa_engine.df[
                    self.wepa_engine.df['{0}_weight'.format(feature)] != 0
                ]) /
                len(self.wepa_engine.df)
            )
    
    def manual_search(self, feature):
        '''
        Will manually loop over weights for a single feature to
        determin the objective best weight
        '''
        ## ensure guesses are in place ##
        if self.best_guesses is None:
            self.best_guesses = []
            for i in self.bounds:
                self.best_guesses.append((i[1]+i[0])/2)
        ## containers for keeping track of best ##
        best_rsq = 0
        best_weight = 0
        ## establish weights and index ##
        idx = self.features.index(feature)
        weights = self.best_guesses.copy()
        ## loop ##
        for weight in range(self.bound[0]*100, self.bound[1]*100+1):
            ## update the value ## 
            weights[idx] = weight / 100
            ## get an rsq ##
            rsqs = self.calc_rsqs(weights)
            if rsqs['wepa'] > best_rsq:
                ## if new best rsq found, update the current best ##
                best_rsq = rsqs['wepa']
                best_weight = weight / 100
        ## return ##
        return best_weight, best_rsq
