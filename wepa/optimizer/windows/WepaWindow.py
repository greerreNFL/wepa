import pandas as pd
import numpy

class WepaWindow():
    '''
    Splits a dataframe into in sample and out of sample
    sets based on specified windowing function
    '''

    def __init__(self, pbp):
        self.pbp = pbp ## can be standard pbp or games df
        self.team_games, self.games = self.get_games(pbp)
        self.df_windowed = None
        self.get_games(pbp)

    def gen_window(self, window_type):
        '''
        Returns a windowed df based on specified type
        '''
        ## lookup ##
        windows = {
            'rand' : self.rand,
            'team_halves' : self.team_halves,
        }
        ## apply ##
        self.df_windowed = windows[window_type]()
        ## return ##
        return self.df_windowed

    def get_games(self, pbp):
        '''
        Aggregates pbp into team games and games
        '''
        ## all team games ##
        team_games = pbp.groupby([
            'game_id' , 'season', 'week', 'posteam'
        ]).agg(
            ## note, posteam margin is added from games.csv by the 
            ## WepaDataHandler. Taking max gives the final game margin
            ## even though this operates over a PBP dataset
            margin = ('posteam_margin', 'max'),
        ).reset_index().rename(columns={
            'posteam' : 'team'
        })
        ## isolate just games ##
        games = team_games[['game_id', 'season', 'week']].copy().drop_duplicates()
        ## return ##
        return team_games, games
    
    #############
    ## Helpers ##
    #############
    def rand_by_id(self, still_oos, season_team_id):
        '''
        For a given season_team id, randomly select a game id that is still oos
        '''
        ## get season and team ##
        season, team = season_team_id.split('_')
        ## select and return 1 game id. If a new 
        filtered = still_oos[
            (still_oos['season'] == int(season)) &
            (still_oos['team'] == team)
        ].copy()
        if len(filtered) > 0:
            return filtered.sample(1).iloc[0]['game_id']
        else:
            ## if a game cant be found, return None which will be handled
            ## by the interpretor of rand_by_id
            return None
    
    def build_season_team_container(self, team_games, init=True):
        '''
        Build a dict of seaosn_team_id's and either init their game count
        to 0 or set it to the total for that team
        '''
        struc = {}
        temp = team_games.groupby(['season', 'team']).agg(
            total_games = ('game_id', 'nunique')
        ).reset_index()
        ## apply the id ##
        temp['season_team'] = (
            temp['season'].apply(str) +
            '_' +
            temp['team']
        )
        for index, row in temp.iterrows():
            ## if initializing, return 0, else give total
            struc[row['season_team']] = 0 if init else row['total_games']
        ## return ##
        return struc

    def build_formed_rand(self, threshold=8):
        '''
        Attempts to build a "formed random" set. This cycles through teams
        and randomly assigns a game as in sample to the team with the fewest
        until all teams meet the threshold
        '''
        ## init a counter to keep track of the lowest number 
        ## of in sample games by a team ##
        lowest_is = 0
        ## init structs to keep track of teams ##
        is_tracker = self.build_season_team_container(self.team_games)
        oos_tracker = self.build_season_team_container(self.team_games, init=False)
        ## init a counter to keep track of the lowest number
        ## of out of sample games by a team ##
        lowest_oos = oos_tracker[min(oos_tracker, key=oos_tracker.get)]
        ## container for games selected as in sample ##
        is_game_ids = []
        ## create a df of out of sample games ##
        still_oos = self.team_games.copy()
        ## loop ##
        while lowest_is < threshold and lowest_oos > threshold-3:
            ## loop until conditions are met ##
            ## get the team with the least is games ##
            lowest_team = min(is_tracker, key=is_tracker.get)
            ## get a game id for them ##
            new_is_game_id = self.rand_by_id(still_oos, lowest_team)
            ## handle error ##
            if new_is_game_id is None:
                print('               Formed random could not find a game')
                print('                    Current lowest  IS: {0}'.format(lowest_is))
                print('                    Current lowest OOS: {0}'.format(lowest_oos))
                return None
            ## add game to new in sample set ##
            is_game_ids.append(new_is_game_id)
            ## get the teams involved in this game and update their trackers
            for index, row in still_oos[
                still_oos['game_id']==new_is_game_id
            ].iterrows():
                lookup = '{0}_{1}'.format(
                    row['season'], row['team']
                )
                is_tracker[lookup] = is_tracker[lookup] + 1
                oos_tracker[lookup] = oos_tracker[lookup] - 1
            ## Remove game from still_oos now that its added to in sample ##
            still_oos = still_oos[still_oos['game_id']!=new_is_game_id].copy()
            ## update set wide values ##
            lowest_is = is_tracker[min(is_tracker, key=is_tracker.get)]
            lowest_oos = oos_tracker[min(oos_tracker, key=oos_tracker.get)]
        ## handle loop end ##
        if lowest_is >= threshold-2 and lowest_oos >= threshold-3:
            ## if thresholds met, check to make sure there is no overlap ##
            potential_overlap = still_oos[numpy.isin(
                still_oos['game_id'],
                is_game_ids
            )]
            if len(potential_overlap) > 0:
                print('               Found in sample game IDs in the out of sample set')
                print(potential_overlap)
                raise Exception('Formed random failed')
            # else, return set of in sample games ##
            return is_game_ids
        else:
            ## if thresholds are not met, it means too many games ##
            ## were removed from out of sample ##
            return None

    #########################
    ## WINDOWING FUNCTIONS ##
    #########################
    def rand(self):
        '''
        Uses the formed random  algo outlined above to attempt to build a set
        of games that give every team at least 6 games in the is and oos
        '''
        counter = 1
        while counter < 11:
            ## get a set of in sample games ##
            in_sample_game_ids = self.build_formed_rand()
            ## if it passes, return, else try again ##
            if in_sample_game_ids is not None:
                self.games['window'] = numpy.where(
                    numpy.isin(
                        self.games['game_id'],
                        in_sample_game_ids
                    ),
                    'in_sample',
                    'out_of_sample'
                )
                return pd.merge(
                    self.team_games,
                    self.games[[
                        'game_id', 'window'
                    ]],
                    on=['game_id'],
                    how='left'
                )
            ## if it does not pass try again ##
            counter += 1
        ## handle error if we got here ##
        raise Exception('Formed random sampling failed after 10 attempts. This is unlikely but possible. Please try again')

    def full_rand(self):
        '''
        Splits df with a completely random selection of games
        This method can lead to under sampling where a team in a
        particular season has a poor split with a low "n" of games in
        the oos or is set
        '''
        ## split games randomly ##
        self.games['window'] = numpy.random.choice(
            ['in_sample', 'out_of_sample'],
            size=len(self.games)
        )
        ## apply to team games ##
        windowed = pd.merge(
            self.team_games,
            self.games[[
                'game_id', 'window'
            ]],
            on=['game_id'],
            how='left'
        )
        return windowed

    def team_halves(self):
        '''
        Splits df based on teams first 8 games
        '''
        ## get game count ##
        windowed = self.team_games.copy()
        windowed['team_game_count'] = windowed.groupby([
            'team', 'season'
        ])['game_id'].cumcount() + 1
        ## window ##
        windowed['window'] = numpy.where(
            windowed['team_game_count'] <= 8,
            'in_sample',
            'out_of_sample'
        )
        ## drop counter ##
        windowed = windowed.drop(columns=[
            'team_game_count'
        ])
        ## return ##
        return windowed

    def season_halves(self):
        '''
        Splits df based on the first 8 weeks
        '''
        ## get game count ##
        windowed = self.team_games.copy()
        ## window ##
        windowed['window'] = numpy.where(
            windowed['week'] <= 8,
            'in_sample',
            'out_of_sample'
        )
        ## return ##
        return windowed

