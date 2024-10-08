import timeit
#from preprocessor.dl_preprocessing import *

# Machine learning module
from ml_logic.preprocessor.ml_preprocessing import load_data, clean_injury_data, clean_playlist_data, engineered_fatigue, merge_df
# from ml_logic.encoders_models.encoders_models import *
from ml_logic.params import *

def main(INJURY_DF,PLAYLIST_DF,TRACKS_DF):
    print('skip to model:')
    preprocessing = input("y or n --> ")
    if preprocessing == "n":
        # TIME
        start = timeit.timeit()

        # load the dfs
        injury_df, playlist_df, tracks_df = load_data(INJURY_DF, PLAYLIST_DF, TRACKS_DF)

        # data format optimization
        #injury_df, playlist_df, tracks_df = transform_format(injury_df, playlist_df, tracks_df)
        #injury_df, playlist_df, tracks_df = optimize_data_type(injury_df, playlist_df, tracks_df)

        # clean the data
        cleaned_injury = clean_injury_df(injury_df)
        cleaned_playlist = clean_playlist_df(playlist_df)
        cleaned_tracks = clean_tracks_df(tracks_df)

        # resize tracks_df
        resized_tracks = resize_tracks_df(cleaned_tracks, RESIZE_STEP)

        # feature engineer
        engineered_injury = engineering_injury_df(cleaned_injury)
        engineered_playlist_key_game = engineering_playlist_df_max_game(cleaned_playlist)
        engineered_playlist_df_weather = engineering_playlist_df_weather(engineered_playlist_key_game, WET)
        engineered_playlist_df_indor = engineering_playlist_df_indor(engineered_playlist_df_weather, INDOOR)
        engineered_playlist_df_temp = engineering_playlist_df_temp(engineered_playlist_df_indor)

        tracks_split = engineering_tracks_df_split(resized_tracks)
        tracks_speed = engineering_tracks_df_true_distance(tracks_split)
        tracks_turn = engineering_tracks_df_turn(tracks_speed)
        tracks_speed = engineering_tracks_df_true_speed(tracks_turn)
        tracks_degree_diff = engineering_tracks_df_degree_diff(tracks_speed)
        tracks_fill_event = engineering_tracks_df_fill_event(tracks_degree_diff)
        tracks_violent_turn = engineering_tracks_df_violent_turn(tracks_fill_event)

        #resize tracks by playkey
        resized_tracks = resize_tracks_df_playkey(tracks_violent_turn)

        # merge the dfs
        merged_df = merge_df(engineered_injury,engineered_playlist_df_temp,resized_tracks)

        # Clean the columns
        engineered_merged_df_fillna = engineering_merged_df_fillna(merged_df)
        cleaned_col = clean_columns(engineered_merged_df_fillna, DELETE_COL)
        end = timeit.timeit()

        # TIME
        print(end - start)
    else:
        print('Start model:')
        preprocessing = input("y or n --> ")
        if preprocessing == "y":

            # Run the pipeline
            dl_pipe, X_train, y_train, X_test, y_test = dl_pipeline(cleaned_col)
            dl_cross_val(dl_pipe, X_train, y_train, X_test, y_test)
        else:
            pass

def ml_model(INJURY_DF,PLAYLIST_DF):
    # Load DataFrames
    injury_data, playlist_data = load_data(INJURY_DF, PLAYLIST_DF)

    # Clean and basic categorization
    clean_injuryData = clean_injury_data(injury_data, playlist_data)
    preclean_playlistData = clean_playlist_data(playlist_data)

    # Feature Engineering
    clean_playlistData = engineered_fatigue(preclean_playlistData)

    # Merge clean_injuryData and clean_playlistData
    mergePlayerKey, mergeGameID, mergePlayKey = merge_df(clean_injuryData, clean_playlistData)

if __name__ == '__main__':
    # execute only if run as the entry point into the program
    #main(INJURY_DF,PLAYLIST_DF,TRACKS_DF)
    ml_model(INJURY_DF,PLAYLIST_DF)
