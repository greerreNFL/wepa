import pandas as pd
import numpy

import pathlib
import os
import random

from .optimizer import WepaOptimizer
from .engine import WepaEngine
from .db import DataHandler

def measure_features(
        features=[], seasons=[], iterations=None,
        rounds=5, combiner='multiplicative',
        combiner_params=None,
        in_run_output=None,
    ):
    '''
    Measures individual feature efficacy seasons by taking iterative semi-random samples
    * This will output a distribution of the optimal weigtht for each feature,
    which is useful to determine how much signal the feature truly has
    * Select all features and seasons by not passing those params
    * If the number of iterations is not set, it will default to the number of seasons
    * Seasons are split into deciles and randomly selected for each iteration,
    meaning each iteration is trained over 10 random seasons
    * Rounds constitutes the number of times an iteration should be resampled
    * The formula for the number of runs per feature is iterations * rounds
    '''
    print('Measuring feature efficacy...')
    ## load data ##
    print('     Loading data...')
    data = DataHandler()
    print('     Determining run sets...')
    ## output structure for results ##
    records = []
    ## set up features ##
    if len(features) == 0:
        ## get all features ##
        ## get path to feature definitions ##
        feature_locations = '{0}/engine/features/definitions'.format(
            pathlib.Path(__file__).parent.resolve()
        )
        ## get all feature names ##
        for file_name in os.listdir(feature_locations):
            ## handle pycache and any other junk in the folder ##
            if not file_name.endswith('.py'):
                continue
            feature_name = file_name[:-3]
            features.append(feature_name)
            features.append('d_{0}'.format(feature_name))
    print('          Found {0} features...'.format(len(features)))
    ## Determine runs ##
    if len(seasons) == 0:
        ## if no seasons provided, set get all from loaded data ##
        for season in range(data.first_completed_season, data.last_completed_season+1):
            seasons.append(season)
    ## determine iterations ##
    iterations = len(seasons) if not isinstance(iterations,int) else iterations
    print('          Found {0} seasons...'.format(len(seasons)))
    print('          Run depth set at {0} rounds'.format(rounds))
    print('          Total optimizations: {0}'.format(
        len(features) * len(seasons) * rounds
    ))
    ## run optis ##
    print('     Running optimizations...')
    for index, feature in enumerate(features):
        print('          Optimizing {0} ({1} of {2})'.format(
            feature, index+1, iterations
        ))
        ## temp recs for in opti updates ##
        feature_recs = []
        for season_round in range(0,iterations):
            ## get a set of seasons ##
            seasons_sampled = []
            ## bin seasons into quartiles ##
            qs = numpy.array_split(seasons,10)
            ## randomly sample quartile ##
            for q in qs:
                numpy.random.choice(q,1)
                seasons_sampled.append(
                    numpy.random.choice(q,1)[0]
                )
            ## get df of seasons ##
            train_df = data.pbp[
                numpy.isin(
                    data.pbp['season'],
                    seasons_sampled
                )
            ].copy()
            ## create counter feature ##
            ## since defense is a gauranteed down weight, 
            ## all offensive weights will have a positive bias
            ## when optimized in isolation. Create a counter feautre
            ## on the other side of the ball to mitigate
            counter_feature = None
            if 'feature'[:2] == 'd_':
                counter_feature = 'plays_all'
            else:
                counter_feature = 'd_plays_all'
            ## init optimizer ##
            optimizer = WepaOptimizer(
                train_df=train_df,
                features=[feature, counter_feature],
                combiner=combiner,
                combiner_params=combiner_params
            )
            for round_num in range(0, rounds):
                if round_num > 0:
                    ## get a new random window for in/out of sample
                    optimizer.rewindow()
                ## optimize ##
                optimizer.optimize()
                ## write to records ##
                rec = {
                    'feature' : feature,
                    'seasons' : ','.join(str(s) for s in seasons_sampled),
                    'round' : round_num,
                    'run_time' : optimizer.opti_rec['run_time'],
                    'iterations' : optimizer.opti_rec['iterations'],
                    'avg_time_per_eval' : optimizer.opti_rec['avg_time_per_eval'],
                    'optimal_weight' : optimizer.opti_rec[feature],
                    'wepa_rsq' : optimizer.opti_rec['wepa_rsq'],
                    'epa_rsq' : optimizer.opti_rec['epa_rsq'],
                    'margin_rsq' : optimizer.opti_rec['margin_rsq'],
                    'lift' : optimizer.opti_rec['lift'],
                    'pct_of_plays' : optimizer.opti_rec['{0}_plays_pct'.format(feature)]
                }
                feature_recs.append(rec)
                records.append(rec)
        ## print in line update on opti ##
        temp_df = pd.DataFrame(feature_recs)
        print('               Run time for all rounds: {0}'.format(
            temp_df['run_time'].sum()
        ))
        print('               Median Optimal Weight: {0}'.format(
            temp_df['optimal_weight'].median()
        ))
        print('               Avg lift: {0}'.format(
            temp_df['lift'].mean()
        ))
        print('               Lift deviation: {0}'.format(
            temp_df['lift'].std()
        ))
        if in_run_output is not None:
            ## if an output path was passed, save ##
            pd.DataFrame(records).to_csv(in_run_output)    ## return a df of optimizations ##
    return pd.DataFrame(records)


def measure_sets(
        features=[], seasons=[], iterations=None,
        combiners=[],rounds=10,
        in_run_output=None,
        with_a_drop=False
    ):
    '''
    Measures the efficiency of a set of features over any combiner passed
    * Seasons and iterations follow the same logic as measure_features, but
    the set of features and combiners must be set
    * In addition to training the model over the semi-random samples, this function
    also measures the results of those optimizations over all data not included in the
    training set (ie a test set)
    * Setting with_a_drop to True will randomly hold an individual feature out
    from an iteration to help determine if it provides signal vs overfitting in
    the test set
    '''
    print('Optimizing feature set...')
    ## load data ##
    print('     Loading data...')
    data = DataHandler()
    print('     Determining run sets...')
    ## output structure for results ##
    records = []
    ## set up features ##
    if len(seasons) == 0:
        ## if no seasons provided, set get all from loaded data ##
        for season in range(data.first_completed_season, data.last_completed_season+1):
            seasons.append(season)
    ## determine iterations ##
    iterations = 6 * len(seasons) if not isinstance(iterations, int) else iterations
    print('          Found {0} seasons...'.format(len(seasons)))
    print('          Run depth set at {0} rounds'.format(rounds))
    print('          Total optimizations: {0}'.format(
        2 * len(seasons) * rounds * len(combiners)
    ))
    ## run optis ##
    print('     Running optimizations...')
    for season_round in range(0,6 * len(seasons)):
        ## get a set of seasons ##
        seasons_sampled = []
        ## bin seasons into quartiles ##
        qs = numpy.array_split(seasons,10)
        ## randomly sample quartile ##
        for q in qs:
            numpy.random.choice(q,1)
            seasons_sampled.append(
                numpy.random.choice(q,1)[0]
            )
        ## get df of seasons ##
        train_df = data.pbp[
            numpy.isin(
                data.pbp['season'],
                seasons_sampled
            )
        ].copy()
        test_df = data.pbp[
            ~numpy.isin(
                data.pbp['season'],
                seasons_sampled
            )
        ].copy()
        ## init optimizer ##
        ## test the combiners ##
        for combiner in combiners:
            ## exlude a random feature if flagged ##
            features_ = features.copy()
            dropped = numpy.nan
            if with_a_drop:
                ## determine if a feature should be dropped ##
                if random.random() <= .5:
                    random.shuffle(features_)
                    dropped = features_.pop()
            optimizer = WepaOptimizer(
                train_df=train_df,
                features=features_,
                combiner=combiner['name'],
                combiner_params=combiner['params']
            )
            ## init optimizer for all game not in the train ##
            optimizer_test = WepaOptimizer(
                train_df=test_df,
                features=features_,
                combiner=combiner['name'],
                combiner_params=combiner['params']
            )
            for round_num in range(0, rounds):
                if round_num > 0:
                    ## get a new random window for in/out of sample
                    optimizer.rewindow()
                    optimizer_test.rewindow()
                ## optimize ##
                optimizer.optimize()
                ## apply to testing ##
                test_rsqs = optimizer_test.calc_rsqs(optimizer.opti_weights)
                ## write to records ##
                rec = {
                    'features' : ','.join(str(s) for s in features),
                    'feature_held_out' : dropped,
                    'combiner' : combiner['name'],
                    'combiner_params' : combiner['params'],
                    'seasons' : ','.join(str(s) for s in seasons_sampled),
                    'round' : round_num,
                    'run_time' : optimizer.opti_rec['run_time'],
                    'iterations' : optimizer.opti_rec['iterations'],
                    'avg_time_per_eval' : optimizer.opti_rec['avg_time_per_eval'],
                    'wepa_rsq' : optimizer.opti_rec['wepa_rsq'],
                    'epa_rsq' : optimizer.opti_rec['epa_rsq'],
                    'margin_rsq' : optimizer.opti_rec['margin_rsq'],
                    'lift' : optimizer.opti_rec['lift'],
                    'lift_pct' : optimizer.opti_rec['wepa_rsq'] / optimizer.opti_rec['epa_rsq'] - 1,
                    'test_lift' : test_rsqs['wepa'] - test_rsqs['epa'],
                    'test_lift_pct' : test_rsqs['wepa'] / test_rsqs['epa'] - 1
                }
                for feature in features:
                    rec['{0}_weight'.format(feature)] = optimizer.opti_rec.get(feature)
                records.append(rec)
                if in_run_output is not None:
                    pd.DataFrame(records).to_csv(in_run_output)
    ## return a df of optimizations ##
    return pd.DataFrame(records)

def measure_optimizer_params(
    test_features=['d_plays_all', 'passes', 'fumble_all', 'qb_rush'],
    rounds=50, seasons=[], in_run_output=None
):
    '''
    Tests different parameters of the optimizer to determine which 
    get closest to the true result
    * This function iterates through different optimizer parameters in 
    shows the result of the optimization for an individual feature against
    it's imperically determined optimial weight
    * The imperical optimal weight is found by testing each possible discount
    at a percision of 2
    '''
    ## define features to test ##
    scales = [1, 3]
    steps = [0.01, 0.001, 0.0001, 0.00001, 0.000001]
    tols = [
        0.0001, 0.00001, 0.000001, 0.0000001,
        0.00000001, 0.000000001
    ]
    methods = ['SLSQP', 'L-BFGS-B']
    ## 
    print('Measuring optimizer parameters...')
    print('     This will run over {0} combinations...'.format(
        len(test_features) * len(scales) * len(steps) * len(tols) * len(methods) * rounds
    ))
    ## load data ##
    print('     Loading data...')
    data = DataHandler()
    ## determine seasons ##
    if len(seasons) == 0:
        ## if no seasons provided, set get all from loaded data ##
        for season in range(data.first_completed_season, data.last_completed_season+1):
            seasons.append(season)
    ## output structure for results ##
    records = []
    ## run optis ##
    print('     Running optimizations...')
    for opti_round in range(0, rounds):
        ## get a set of seasons ##
        seasons_sampled = []
        ## bin seasons into quartiles ##
        qs = numpy.array_split(seasons,4)
        ## randomly sample quartile ##
        for q in qs:
            numpy.random.choice(q,1)
            seasons_sampled.append(
                numpy.random.choice(q,1)[0]
            )
        ## get df of seasons ##
        train_df = data.pbp[
            numpy.isin(
                data.pbp['season'],
                seasons_sampled
            )
        ].copy()
        ## init optimizer ##
        optimizer = WepaOptimizer(
            train_df=train_df,
            features=[test_features[0]],
            combiner='mean',
            combiner_params={}
        )
        ## test each model param ##
        for test_feature in test_features:
            ## update optimizer ##
            optimizer.update_features([test_feature])
            ## establish objective best weight for feature ##
            objective_weight, objective_rsq = optimizer.manual_search(test_feature)
            for scale in scales:
                for step in steps:
                    for tol in tols:
                        for method in methods:
                            ## update optimizer ##
                            optimizer.scale=scale
                            optimizer.step=step
                            optimizer.tol=tol
                            optimizer.method=method
                            ## optimize ##
                            optimizer.optimize()
                            ## write to records ##
                            rec = {
                                'feature' : test_feature,
                                'seasons' : ','.join(str(s) for s in seasons_sampled),
                                'round' : opti_round,
                                'scale' : scale,
                                'step' : step,
                                'tol' : tol,
                                'method' : method,
                                'run_time' : optimizer.opti_rec['run_time'],
                                'iterations' : optimizer.opti_rec['iterations'],
                                'avg_time_per_eval' : optimizer.opti_rec['avg_time_per_eval'],
                                'objective_weight' : objective_weight,
                                'objective_rsq' : objective_rsq,
                                'optimal_weight' : optimizer.opti_rec[test_feature],
                                'wepa_rsq' : optimizer.opti_rec['wepa_rsq'],
                                'epa_rsq' : optimizer.opti_rec['epa_rsq'],
                                'margin_rsq' : optimizer.opti_rec['margin_rsq'],
                                'lift' : optimizer.opti_rec['lift'],
                                'pct_of_plays' : optimizer.opti_rec['{0}_plays_pct'.format(test_feature)]
                            }
                            records.append(rec)
                            if in_run_output is not None:
                                ## if an output path was passed, save ##
                                pd.DataFrame(records).to_csv(in_run_output) ## return a df of optimizations ##
    ## print in line update on opti ##
    return pd.DataFrame(records)


def get_applied_wepa(
    features, weights=None, df=None,
    combiner='multiplicative',
    combiner_params=None,
):
    '''
    Helper function for calc_overlaps that applies a .5 weight to all
    features passed. This is meant to determine the % of plays for which
    any two features are both active
    '''
    if df is None:
        data = DataHandler()
        df = data.pbp
    ## init engine ##
    engine = WepaEngine(
        df=df,
        features=features,
        weights=[.5]*len(features) if weights is None else weights,
        combiner=combiner,
        combiner_params=combiner_params
    )
    ## apply ##
    engine.apply_wepa()
    return engine.df

def calc_overlaps(
    features
):
    '''
    Calculates how much features overlap with each other
    '''
    ## init a wepa pbp with all active weights at .5 (the default) ##
    df = get_applied_wepa(features=features)
    ## struc ##
    records=[]
    ## calc the overlaps ##
    for f1 in features:
        for f2 in features:
            ## determine types
            f1_type = 'off' if f1[:2] != 'd_' else 'def'
            f2_type = 'off' if f2[:2] != 'd_' else 'def'
            if f1!=f2 and f1_type == f2_type:
                f1_only = df[df['{0}_weight'.format(f1)]==.5].copy()
                f2_only = df[df['{0}_weight'.format(f2)]==.5].copy()
                combo = df[
                    (df['{0}_weight'.format(f1)]==.5) &
                    (df['{0}_weight'.format(f2)]==.5)
                ].copy()
                records.append({
                    'feature_1' : f1,
                    'feature_2' : f2,
                    'f2_coverage_of_f1' : len(combo)/len(f1_only),
                    'f1_coverage_of_f2' : len(combo)/len(f2_only),
                    'f1_only_avg_epa' : f1_only['epa'].mean(),
                    'f2_only_avg_epa' : f2_only['epa'].mean(),
                    'overlap_avg_epa' : combo['epa'].mean(),
                })
    return pd.DataFrame(records)



def feature_search(
        features=[], seasons=[],rounds=100,
        in_run_output=None,
    ):
    '''
    Progressively Drops features until test set is made worse
    * Use cautiously (read: dont use it) as passing a ton of features
    will cause an explosion in run time. A good approach in theory, terrible
    in practice
    '''
    print('Optimizing feature set...')
    ## load data ##
    print('     Loading data...')
    data = DataHandler()
    print('     Determining run sets...')
    ## output structure for results ##
    records = []
    ## set up features ##
    if len(seasons) == 0:
        ## if no seasons provided, set get all from loaded data ##
        for season in range(data.first_completed_season, data.last_completed_season+1):
            seasons.append(season)
    ## feature search ##
    print('     Running optimizations...')
    ## var to contain results of worst feature ##
    lowest_lift = -1
    while lowest_lift < 0 and len(features) > 1:
        print('          Features Remaining: {0}'.format(features))
        ## container of results ##
        round_records = []
        ## get a set of seasons ##
        seasons_sampled = []
        ## bin seasons into quartiles ##
        qs = numpy.array_split(seasons,10)
        ## randomly sample quartile ##
        for q in qs:
            numpy.random.choice(q,1)
            seasons_sampled.append(
                numpy.random.choice(q,1)[0]
            )
        ## get df of seasons ##
        train_df = data.pbp[
            numpy.isin(
                data.pbp['season'],
                seasons_sampled
            )
        ].copy()
        test_df = data.pbp[
            ~numpy.isin(
                data.pbp['season'],
                seasons_sampled
            )
        ].copy()
        ## cycle features ##
        for feature in features:
            ## exlude the feature
            features_ = features.copy()
            features_.pop(features_.index(feature))
            optimizer = WepaOptimizer(
                train_df=train_df,
                features=features_,
                combiner='multiplicative',
                combiner_params={}
            )
            ## init optimizer for all game not in the train ##
            optimizer_test = WepaOptimizer(
                train_df=test_df,
                features=features_,
                combiner='multiplicative',
                combiner_params={}
            )
            for round_num in range(0, rounds):
                if round_num > 0:
                    ## get a new random window for in/out of sample
                    optimizer.rewindow()
                    optimizer_test.rewindow()
                ## optimize ##
                optimizer.optimize()
                ## apply to testing ##
                test_rsqs = optimizer_test.calc_rsqs(optimizer.opti_weights)
                ## write to records ##
                rec = {
                    'features' : ','.join(str(s) for s in features_),
                    'feature_held_out' : feature,
                    'run_time' : optimizer.opti_rec['run_time'],
                    'iterations' : optimizer.opti_rec['iterations'],
                    'avg_time_per_eval' : optimizer.opti_rec['avg_time_per_eval'],
                    'wepa_rsq' : optimizer.opti_rec['wepa_rsq'],
                    'epa_rsq' : optimizer.opti_rec['epa_rsq'],
                    'margin_rsq' : optimizer.opti_rec['margin_rsq'],
                    'lift' : optimizer.opti_rec['lift'],
                    'lift_pct' : optimizer.opti_rec['wepa_rsq'] / optimizer.opti_rec['epa_rsq'] - 1,
                    'test_lift' : test_rsqs['wepa'] - test_rsqs['epa'],
                    'test_lift_pct' : test_rsqs['wepa'] / test_rsqs['epa'] - 1
                }
                round_records.append(rec)
        ## determine worst feature ##
        df = pd.DataFrame(round_records)
        agg = df.groupby(['feature_held_out']).agg(
            avg_test_lift = ('test_lift_pct', 'mean')
        ).reset_index()
        agg = agg.sort_values(
            by=['avg_test_lift'],
            ascending=[False]
        ).reset_index(drop=True)
        worst_feature = agg.iloc[0]['feature_held_out']
        worst_test_lift = agg.iloc[0]['test_lift_pct']
        ## update worst feature ##
        lowest_lift = df['test_lift_pct'].mean() - worst_test_lift
        ## create record ##
        records.append({
            'features' : ', '.join(str(s) for s in features) if len(features) < 20 else 'too long',
            'num_features' : len(features),
            'worst_feature' : worst_feature,
            'worst_feature_exclusion_lift' : worst_test_lift - df['test_lift_pct'].mean(),
            'total_run_time' : df['run_time'].sum(),
            'avg_run_time' : df['run_time'].mean(),
            'avg_lift' : df['lift_pct'].mean(),
            'avg_test_lift' : df['test_lift_pct'].mean()
        })
        ## drop the worst feature ##
        features.pop(features.index(worst_feature))
        print('               > Feature Dropped: {0}'.format(worst_feature))
        ## output ##
        if in_run_output is not None:
            pd.DataFrame(records).to_csv(in_run_output)
    ## return a df of optimizations ##
    return pd.DataFrame(records)

