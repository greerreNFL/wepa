import pandas as pd 
import numpy
import pathlib

import nfelodcm as dcm

class DataHandler():
    '''
    Loads and stores all data
    '''
    def __init__(self, **kwargs):
        self.pbp = kwargs.get('pbp', None)
        self.games = kwargs.get('games', None)
        self.data_folder = self.set_data_folder()
        self.weights = None
        self.wepa_by_game_at = None
        self.wepa_by_season_at = None
        self.wepa_by_game_pit = None
        self.wepa_by_season_pit = None
        self.first_completed_season = None
        self.last_completed_season = None
        ## retrieve data if not passed ##
        self.load_data()
        self.format_data()
        self.get_season_range()

    def set_data_folder(self):
        '''
        Sets the absolute path for where data is located
        '''
        return '{0}/data'.format(
            pathlib.Path(__file__).parent.parent.parent.resolve()
        )

    def load_data(self):
        '''
        Checks what data was passed and what needs to be pulled
        '''
        ## determine what data is needed ##
        tables_needed = []
        for table_name, table_prop in {
            'pbp' : self.pbp,
            'games' : self.games
        }.items():
            if table_prop is None:
                tables_needed.append(table_name)
        ## get missing ##
        if len(tables_needed) > 0:
            print('     Data was not passed. Retrieving fresh data...')
            db = dcm.load(tables_needed)
            ## update ##
            self.pbp = db.get('pbp', self.pbp)
            self.games = db.get('games', self.games)
        ## remove playoff games ##
        self.games = self.games[
            self.games['game_type'] == 'REG'
        ].copy()
    
    def save_wepa(self):
        '''
        Saves wepa files to the data folder
        '''
        frames = [
            {'name':'WEPA by Game AT', 'df':self.wepa_by_game_at},
            {'name':'WEPA by Game PIT', 'df':self.wepa_by_game_pit},
            {'name':'WEPA by Season AT', 'df':self.wepa_by_season_at},
            {'name':'WEPA by Season PIT', 'df':self.wepa_by_season_pit},
        ]
        for frame in frames:
            if frame['df'] is None:
                print('     {0} dataframe not found and could not save'.format(
                    frame['name']
                ))
            else:
                frame['df'].to_csv(
                    '{0}/{1}.csv'.format(
                        self.data_folder,
                        frame['name'].replace(' ', '_').lower()
                    )
                )

    def format_data(self):
        '''
        Aditional transformations not in dcm load
        '''
        ## determine if game has been played ##
        self.games['is_played'] = numpy.where(
            ~pd.isnull(self.games['result']),
            1,
            0
        )
        ## add home and away margin ##
        self.games['home_margin'] = self.games['result']
        self.games['away_margin'] = self.games['result'] * -1
        ## flatten 
        flat = pd.concat([
            self.games[['game_id', 'home_team', 'home_margin']].rename(columns={
                'home_team' : 'team',
                'home_margin' : 'margin'
            }),
            self.games[['game_id', 'away_team', 'away_margin']].rename(columns={
                'away_team' : 'team',
                'away_margin' : 'margin'
            })
        ])
        ## add to pbp ##
        self.pbp = pd.merge(
            self.pbp,
            flat[['game_id', 'team', 'margin']].rename(columns={
                'team' : 'posteam',
                'margin' : 'posteam_margin'
            }),
            on=['game_id', 'posteam'],
            how='left'
        )
        self.pbp = pd.merge(
            self.pbp,
            flat[['game_id', 'team', 'margin']].rename(columns={
                'team' : 'defteam',
                'margin' : 'defteam_margin'
            }),
            on=['game_id', 'defteam'],
            how='left'
        )
        ## remove any games with no margin ##
        ## this is how playoffs are removed ##
        self.pbp = self.pbp[
            ~pd.isnull(self.pbp['posteam_margin'])
        ].copy()
    
    def get_season_range(self):
        '''
        Gets the first and last season in the data set
        '''
        ## get pct played by season
        seasons = self.games.groupby(['season']).agg(
            pct_played = ('is_played', 'mean')
        ).reset_index()
        ## updated completed seasons
        self.first_completed_season=seasons[
            seasons['pct_played']==1
        ]['season'].min()
        self.last_completed_season=seasons[
            seasons['pct_played']==1
        ]['season'].max()
    
