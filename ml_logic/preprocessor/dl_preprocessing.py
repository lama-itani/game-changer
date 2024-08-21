import pandas as pd
import os

from params import *

def load_data(INJURY_DF,PLAYLIST_DF,TRACKS_DF):
    injury_path = os.path.join(DIR_PATH,DATA_FILE,INJURY_DF)
    playlist_path = os.path.join(DIR_PATH,DATA_FILE,PLAYLIST_DF)
    tracks_path = os.path.join(DIR_PATH,DATA_FILE,TRACKS_DF)
    injury_df = pd.read_csv(injury_path)
    playlist_df = pd.read_csv(playlist_path)
    tracks_df = pd.read_csv(tracks_path)
    return injury_df, playlist_df, tracks_df

def clean_injury_df(injury_df):
    # injury duration column added ooo
    #injury_df['injury_duration'] = injury_df[['DM_M1','DM_M7','DM_M28','DM_M42']].sum(axis=1)

    # ----FILLNA ---- create a playkeyID column with playkey na filled with mode
    # TODO --> MAKE POSSIBLE TO IMPLEMENT OTHER FILL THAN MODE IN ARGUMENT
    injury_df['PlayKeyID'] = injury_df['PlayKey'].str.split("-").str[2]
    injury_df['PlayKeyID'].fillna(injury_df['PlayKeyID'].mode()[0], inplace=True)
    injury_df['new_PlayKey'] = injury_df['GameID']+"-"+injury_df['PlayKeyID']
    return injury_df

def clean_playlist_df(playlist_df):
    # ----FILLNA ---- With mode for stadium type, weather & playtype
    playlist_df['StadiumType'].fillna(playlist_df['StadiumType'].mode()[0], inplace=True)
    playlist_df['Weather'].fillna(playlist_df['Weather'].mode()[0], inplace=True)
    playlist_df['PlayType'].fillna(playlist_df['PlayType'].mode()[0], inplace=True)
    return playlist_df

def engineering_injury_df(injury_df):
    # Add injury_duration column
    injury_df['injury_duration'] = injury_df[['DM_M1','DM_M7','DM_M28','DM_M42']].sum(axis=1)
    return injury_df

def engineering_playlist_df(playlist_df):
    # Add max_game column : max number of game for each player
    playlist_df['max_game'] = playlist_df['PlayKey'].str.split("-").str[1]
    playlist_df['max_game']=playlist_df['max_game'].astype(int)
    playlist_df['max_game'] = (playlist_df.sort_values(['PlayKey']).groupby(['PlayerKey'])['PlayerGame']
                   .transform(lambda x: x.max()))

    # Add total_playkey column : total_playkey played during a game
    playlist_df['total_playkey'] = playlist_df['PlayKey'].str.split("-").str[2]
    playlist_df['total_playkey']=playlist_df['total_playkey'].astype(int)
    playlist_df['playkey_max'] = (playlist_df.sort_values(['PlayKey']).groupby(['PlayerKey','GameID'])['total_playkey']
                   .transform(lambda x: x.max()))

    # Add playkey_total column : sum of all playkeys played during all games
    playlist_df['playkey_total'] = (playlist_df.sort_values(['PlayKey']).groupby(['PlayerKey'])['total_playkey']
                   .transform(lambda x: x.count()))
    return playlist_df

def merge_df(injury_df,playlist_df,tracks_df):
