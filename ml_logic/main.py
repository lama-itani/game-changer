from proprocessor.dl_preprocessing import *
def main(INJURY_DF,PLAYLIST_DF,TRACKS_DF):
    # load the dfs
    injury_df, playlist_df, tracks_df = load_data(INJURY_DF, PLAYLIST_DF, TRACKS_DF)

    # clean the data
    clean_injury_df(injury_df)
    clean_playlist_df(playlist_df)

    # feature engineer
    engineering_injury_df(injury_df)
    engineering_playlist_df(playlist_df)

    # merge the dfs

if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main(INJURY_DF,PLAYLIST_DF,TRACKS_DF)
