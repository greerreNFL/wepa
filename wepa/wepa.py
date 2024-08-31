import pandas as pd
import numpy

import pathlib
import json

from .engine import WepaEngine
from .optimizer import MarginNormalization
from .db import DataHandler
from .weights import WepaWeights

class WepaRunner():
    '''
    Client for accessing all wepa features and data
    * Creating a WepaRunner will:
    *** Load the conig
    *** Check that weights are set for all available seasons
    *** Train weights if they dont exist for all available seasons
    *** Generate the wepa files
    '''
    def __init__(self, **kwargs):
        ## load config
        self.config = self.load_config()
        ## load data ##
        self.data = DataHandler(
            pbp=kwargs.get('pbp', None),
            games=kwargs.get('games', None),
            exclude_playoffs=False
        )
        ## load and check for updated weights ##
        self.weights = WepaWeights(
            self.data,
            self.config
        )
        at_weights, pit_weights = self.weights.get_weights()
        ## init engines ##
        self.at_engine = WepaEngine(
            df=self.data.pbp,
            features=self.config['features'],
            weights=at_weights,
            combiner=self.config['combiner'],
            combiner_params=self.config['combiner_params']
        )
        self.pit_engine = WepaEngine(
            ## ignore first season, which has no pit data
            df=self.data.pbp[
                self.data.pbp['season'] > self.data.first_completed_season
            ].copy(),
            features=self.config['features'],
            weights=pit_weights,
            combiner=self.config['combiner'],
            combiner_params=self.config['combiner_params']
        )
        ## apply wepa ##
        self.at_engine.apply_wepa()
        self.pit_engine.apply_wepa()
        self.save_wepas()
        ## update normalization model ##
        mn = MarginNormalization()
        mn.update_config()

    
    def load_config(self):
        '''
        Loads the config file
        '''
        with open('{0}/model_config.json'.format(
            pathlib.Path(__file__).parent.parent.resolve()
        )) as f:
            return json.load(f)

    def flatten_wepa(self, pbp_df):
        '''
        Takes an applied wepa pbp and aggs + flattens by team
        '''
        agg = pbp_df.groupby([
             'game_id', 'posteam', 'defteam', 'season'
        ]).agg(
            margin = ('posteam_margin', 'max'), ## pre added by data handler
            epa = ('epa', 'sum'),
            wepa = ('wepa_off', 'sum'),
            d_wepa = ('wepa_def', 'sum')
        ).reset_index()
        ## merge ##
        merged = pd.merge(
            agg.rename(columns={
                'posteam' : 'team',
                'defteam' : 'opponent',
                'd_wepa' : 'd_wepa_against'
            }),
            agg.rename(columns={
                'defteam' : 'team',
                'posteam' : 'opponent',
                'margin' : 'margin_against',
                'epa' : 'epa_against',
                'wepa' : 'wepa_against'
            }),
            on=['game_id', 'team', 'opponent', 'season'],
            how='left'
        )
        ## calculate nets ##
        merged['epa_net'] = merged['epa'] - merged['epa_against']
        merged['epa_net_opponent'] = merged['epa_against'] - merged['epa']
        merged['wepa_net'] = merged['wepa'] - merged['d_wepa']
        merged['wepa_net_opponent'] = merged['wepa_against'] - merged['d_wepa_against']
        ## add a game number ##
        merged = merged.sort_values(
            by=['team', 'game_id'],
            ascending=[True, True]
        ).reset_index(drop=True)
        merged['game_number'] = merged.groupby(['team', 'season']).cumcount() + 1
        ## sort columns ##
        merged = merged[[
            'game_id', 'team', 'opponent', 'season',
            'game_number', 'margin', 'margin_against',
            'epa', 'epa_against', 'epa_net',
            'epa_net_opponent', 'wepa', 'd_wepa', 'wepa_net',
            'wepa_against', 'd_wepa_against', 'wepa_net_opponent'
        ]].copy()
        ## return ##
        return merged

    def agg_by_season(self, flat_df):
        '''
        Aggregates a flattened wepa game agg into a season agg
        '''
        return flat_df.drop(columns=[
            'game_id', 'game_number', 'opponent'
        ]).groupby(
            ['season', 'team']
        ).sum().reset_index().sort_values(
            by=['wepa_net'],
            ascending=[False]
        ).reset_index(drop=True)

    def save_wepas(self):
        '''
        Applies wepa and generates wepa files. Saves to data folder
        '''
        ## flatten, agg, and load the wepas in data
        self.data.wepa_by_game_at = self.flatten_wepa(
            self.at_engine.df
        )
        self.data.wepa_by_season_at = self.agg_by_season(self.flatten_wepa(
            self.at_engine.df
        ))
        self.data.wepa_by_game_pit = self.flatten_wepa(
            self.pit_engine.df
        )
        self.data.wepa_by_season_pit = self.agg_by_season(self.flatten_wepa(
            self.pit_engine.df
        ))
        ## normalize ##
        m = self.config['normalization']['m']
        b = self.config['normalization']['b']
        for df in [
            self.data.wepa_by_game_at, self.data.wepa_by_season_at,
            self.data.wepa_by_game_pit, self.data.wepa_by_season_pit
        ]:
            df['wepa_net_normalized'] = (
                df['wepa_net'] * m +
                b
            )
        ## save the data
        self.data.save_wepa()
        

