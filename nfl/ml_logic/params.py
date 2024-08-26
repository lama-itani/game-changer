import os

DIR_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_FILE = 'data/'

INJURY_DF = 'InjuryRecord.csv'
PLAYLIST_DF = 'PlayList.csv'
TRACKS_DF = 'PlayerTrackData.csv'

RESIZE_STEP = 1

WET = {'Clear and warm':0,
 'Mostly Cloudy':0,
 'Sunny':0,
 'Clear':0,
 'Cloudy':0,
 'Cloudy, fog started developing in 2nd quarter':1,
 'Rain':1,
 'Partly Cloudy':0,
 'Mostly cloudy':0,
 'Cloudy and cold':0,
 'Cloudy and Cool':0,
 'Rain Chance 40%':0,
 'Controlled Climate':0,
 'Sunny and warm':0,
 'Partly cloudy':0,
 'Clear and Cool':0,
 'Clear and cold':0,
 'Sunny and cold':0,
 'Light Rain':1,
 'Partly clear':0,
 'Hazy':1,
 'Mostly Sunny':0}

INDOOR = {'Outdoor':0,
 'Indoors':1,
 'Oudoor':0,
 'Outdoors':0,
 'Open':0,
 'Retractable Roof':0,
 'Domed, open':0,
 'Retr. Roof-Closed':1}

# https://webpages.uidaho.edu/~renaes/251/HON/Student%20PPTs/Avg%20NFL%20ht%20wt.pdf
PLAYER = {
    'QB': [75.43,224.97],
    'RB':[70.73,214,48],
    'WR':[72.40,200.32],
    'TE':[76.54,254.26]
}

TRACKS_AGG = {
'time':'max',
'event':'first',
'x':'mean',
'y':'mean',
'dir':'mean',
'dis':'mean',
'o':'mean',
's':'mean',
'PlayerKey':'first',
'GameKey':'first',
'PlayKeyID':'first',
'dist_x':'mean',
'dist_y':'mean',
'true_dist':'sum',
'turn':'mean',
'turn_agg':'max',
'true_speed':'max',
'dir_o_diff':'max',
'45_turn':'count',
'180_turn':'count',
'cumsum_45':'max',
'cumsum_180':'max',
}

DELETE_COL = ["PlayKey","y", "dir", "dis", "o","s",
                "PlayKeyID_x","PlayerKey_tracks", "GameID",
                "PlayerKey_injury", "GameID_injury",
                "PlayKey_injury","DM_M1", "DM_M7", "DM_M28",
                "DM_M42","PlayKeyID_tracks","Surface","PositionGroup"]

# For ML
COLUMNS_TO_DROP = ["PlayerKey", "GameID", "PlayKey",
                    "RosterPosition", "PlayerGame", "Weather",
                    "StadiumType","PlayerGamePlay", "PositionGroup"]
