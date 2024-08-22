import numpy as np
import pandas as pd
import os

from params import *

# Helper function
# https://www.quora.com/What-temperature-is-cold-in-Fahrenheit
def temp_classification(x):
  if x >= 75:
    return 10
  elif x >= 65:
    return 9
  elif x >= 55:
    return 8
  elif x >= 45:
    return 7
  elif x >= 35:
    return 6
  elif x >= 25:
    return 5
  elif x >= 15:
    return 4
  elif x >= 5:
    return 3
  elif x >= -5:
    return 2
  else:
    return 1

# Write to CSV
def write_to_csv(df, file_name):
    df.to_csv(file_name+'.csv',mode='w+' )

# Resize DF
def resize_tracks_df(track_df, step):
    track_df = track_df.iloc[::step, :]
    print("===== resized tracks ======")
    print(track_df.head(1))
    write_to_csv(track_df, 'resize')
    return track_df


# Load data
def load_data(INJURY_DF,PLAYLIST_DF,TRACKS_DF):
    injury_path = os.path.join(DIR_PATH,DATA_FILE,INJURY_DF)
    playlist_path = os.path.join(DIR_PATH,DATA_FILE,PLAYLIST_DF)
    tracks_path = os.path.join(DIR_PATH,DATA_FILE,TRACKS_DF)
    injury_df = pd.read_csv(injury_path)
    playlist_df = pd.read_csv(playlist_path)
    tracks_df = pd.read_csv(tracks_path)
    return injury_df, playlist_df, tracks_df

# Transform to parquet
def transform_format(injury_df, playlist_df, tracks_df):
    injury_df.to_parquet("cleaned_injury.parquet")
    playlist_df.to_parquet("cleaned_playlist.parquet")
    tracks_df.to_parquet("cleaned_tracks.parquet")

    injury_df = pd.read_parquet("cleaned_injury.parquet")
    playlist_df = pd.read_parquet("cleaned_playlist.parquet")
    tracks_df = pd.read_parquet("cleaned_tracks.parquet")
    return injury_df, playlist_df, tracks_df

# Optimize datatype
def optimize_data_type(injury_df, playlist_df, tracks_df):
    injury_df.select_dtypes(include='object').astype("category")
    playlist_df.select_dtypes(include='object').astype("category")
    tracks_df.select_dtypes(include='object').astype("category")
    return injury_df, playlist_df, tracks_df

# Clean data
def clean_injury_df(injury_df):
    # ----FILLNA ---- create a playkeyID column with playkey na filled with mode
    # TODO --> MAKE POSSIBLE TO IMPLEMENT OTHER FILL THAN MODE IN ARGUMENT
    injury_df['PlayKeyID'] = injury_df['PlayKey'].str.split("-").str[2]
    injury_df['PlayKeyID'].fillna(injury_df['PlayKeyID'].mode()[0], inplace=True)
    injury_df['new_PlayKey'] = injury_df['GameID']+"-"+injury_df['PlayKeyID']
    print("===== injury_df FILLNA ======")
    print(injury_df.head(1))
    return injury_df

def clean_playlist_df(playlist_df):
    # ----FILLNA ---- With mode for stadium type, weather & playtype
    playlist_df['StadiumType'].fillna(playlist_df['StadiumType'].mode()[0], inplace=True)
    playlist_df['Weather'].fillna(playlist_df['Weather'].mode()[0], inplace=True)
    playlist_df['PlayType'].fillna(playlist_df['PlayType'].mode()[0], inplace=True)
    print("===== playlist_df FILLNA ======")
    print(playlist_df.head(1))
    return playlist_df

def clean_tracks_df(track_df):
    # ----FILLNA ---- Only 2 in o and dir col. Replaced with median for ballsnap event (the two occurences are ball-snaps)
    # with mean to account for outliers for o and dir
    try:
        nan_ballsnap = track_df.loc[track_df.event=='ball_snap']
        mean_o_ballsnap = round(nan_ballsnap.o.mean(),2)
        mean_dir_ballsnap = round(nan_ballsnap.dir.mean(),2)
        track_df['o'].loc[45693011] = mean_o_ballsnap
        track_df['dir'].loc[45693011] = mean_dir_ballsnap
        track_df['o'].loc[22184114] = mean_o_ballsnap
        track_df['dir'].loc[22184114] = mean_dir_ballsnap
        print("===== clean_tracks_df FILLNA ======")
        print(track_df.head(1))
        return track_df
    except:
        return track_df

# Engineer data
# Injury
def engineering_injury_df(injury_df):
    # Add injury_duration column
    injury_df['injury_duration'] = injury_df[['DM_M1','DM_M7','DM_M28','DM_M42']].sum(axis=1)
    print("===== engineering_injury_df ======")
    print(injury_df.head(1))
    return injury_df

# Playlist
def engineering_playlist_df_max_game(playlist_df):
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
    print("===== engineering_playlist_df ======")
    print(playlist_df.head(1))
    return playlist_df

def engineering_playlist_df_weather(playlist_df, dict):
    playlist_df['Wet'] = playlist_df['Weather'].map(dict)
    print("===== engineering_playlist_df WEATHER ======")
    print(playlist_df.head(1))
    return playlist_df

def engineering_playlist_df_indor(playlist_df, dict):
    playlist_df['Indoor'] = playlist_df['StadiumType'].map(dict)
    print("===== engineering_playlist_df Indoor ======")
    print(playlist_df.head(1))
    return playlist_df

def engineering_playlist_df_temp(playlist_df):
    playlist_df['Temperature_transformed'] = playlist_df['Temperature'].map(temp_classification)
    print("===== engineering_playlist_df Temperature ======")
    print(playlist_df.head(1))
    return playlist_df

# Tracks
def engineering_tracks_df_split(track_df):
    # Split playkey col for further taks into game Id, player Id, playkey Id
    track_df['PlayerKey'] = track_df['PlayKey'].str.split("-").str[0]
    track_df['GameKey'] = track_df['PlayKey'].str.split("-").str[1]
    track_df['PlayKeyID'] = track_df['PlayKey'].str.split("-").str[2]
    print("===== Split ======")
    print(track_df.head(1))
    return track_df

def engineering_tracks_df_true_distance(track_df):
    # Compute true distance
    track_df['dist_x'] = (track_df.sort_values(['PlayKey','time']).groupby(['PlayerKey', 'GameKey'])['x']
                   .transform(lambda x: round(x.diff(),2)))
    track_df['dist_y'] = (track_df.sort_values(['PlayKey','time']).groupby(['PlayerKey', 'GameKey'])['y']
                    .transform(lambda x: round(x.diff(),2)))
    track_df['true_dist'] = round(np.sqrt(track_df['dist_x']**2 + track_df['dist_y']**2),2)
    print("===== true distance ======")
    print(track_df.head(1))
    return track_df

def engineering_tracks_df_turn(track_df):
    # Turn and turn aggregation
    track_df['turn'] = (track_df.sort_values(['PlayKey','time']).groupby(['PlayerKey', 'GameKey'])['dir']
                   .transform(lambda x: round(x.diff(),2)))
    track_df['turn_agg'] = (track_df.sort_values(['PlayKey','time']).groupby(['PlayerKey', 'GameKey'])['turn']
                   .transform(lambda x: round(x.rolling(5, min_periods=1).sum(),2)))

    print("===== Turn ======")
    print(track_df.head(1))
    return track_df

def engineering_tracks_df_true_speed(track_df):
    # True speed
    track_df['true_speed'] = round(track_df['true_dist']/track_df['time'],2)
    print("===== speed ======")
    print(track_df.head(1))
    return track_df

def engineering_tracks_df_degree_diff(track_df):
    # degree diff between orientation and direction
    track_df['dir_o_diff'] = round(track_df['dir']-track_df['o'],2)
    print("===== degrre diff ======")
    print(track_df.head(1))
    return track_df

def engineering_tracks_df_fill_event(track_df):
    # Fill events
    track_df['event'].ffill(inplace=True)
    print("===== fill event ======")
    print(track_df.head(1))
    return track_df

def engineering_tracks_df_violent_turn(track_df):
    # violent turns
    track_df['45_turn'] = track_df['turn'].apply(lambda x : 1 if(x>45 or x<-45) else 0)
    track_df['180_turn'] = track_df['turn'].apply(lambda x : 1 if(x>180 or x<-180) else 0)

    # Nb of violent turns
    track_df['cumsum_45'] = (track_df.sort_values(['PlayKey','time']).groupby(['PlayerKey', 'GameKey'])['45_turn']
                   .transform(lambda x: x.cumsum()))
    track_df['cumsum_180'] = (track_df.sort_values(['PlayKey','time']).groupby(['PlayerKey', 'GameKey'])['180_turn']
                   .transform(lambda x: x.cumsum()))

    print("===== violent turns ======")
    print(track_df.head(1))
    return track_df

# Merge the DFs
def merge_df(injury_df,playlist_df,tracks_df):
    merged_playlist_injury = playlist_df.merge(how='left', right=injury_df, left_on='PlayKey',right_on='new_PlayKey',suffixes=('', '_injury'))
    merged_playlist_injury_tracks = tracks_df.merge(how='left', right=merged_playlist_injury, on='PlayKey',suffixes=('_x', '_tracks'))
    print(merged_playlist_injury_tracks.head(1))
    write_to_csv(merged_playlist_injury_tracks, 'final')
    return merged_playlist_injury_tracks
