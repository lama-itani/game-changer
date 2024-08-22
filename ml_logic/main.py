from preprocessor.dl_preprocessing import *
def main(INJURY_DF,PLAYLIST_DF,TRACKS_DF):
    # load the dfs
    injury_df, playlist_df, tracks_df = load_data(INJURY_DF, PLAYLIST_DF, TRACKS_DF)

    # clean the data
    cleaned_injury = clean_injury_df(injury_df).head(10000)
    cleaned_playlist = clean_playlist_df(playlist_df).head(10000)
    cleaned_tracks = clean_tracks_df(tracks_df).head(10000)

    # resize tracks_df
    resized_tracks = resize_tracks_df(cleaned_tracks, RESIZE_STEP)

    # feature engineer
    engineered_injury = engineering_injury_df(cleaned_injury)
    engineered_playlist = engineering_playlist_df(cleaned_playlist)

    tracks_split = engineering_tracks_df_split(resized_tracks)
    tracks_speed = engineering_tracks_df_true_distance(tracks_split)
    tracks_turn = engineering_tracks_df_turn(tracks_speed)
    tracks_speed = engineering_tracks_df_true_speed(tracks_turn)
    tracks_degree_diff = engineering_tracks_df_degree_diff(tracks_speed)
    tracks_fill_event = engineering_tracks_df_fill_event(tracks_degree_diff)
    tracks_violent_turn = engineering_tracks_df_violent_turn(tracks_fill_event)
    # merge the dfs
    #merge_df(injury_df,playlist_df,tracks_df)

if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main(INJURY_DF,PLAYLIST_DF,TRACKS_DF)
