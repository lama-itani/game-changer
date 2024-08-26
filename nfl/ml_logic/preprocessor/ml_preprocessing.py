import numpy as np
import pandas as pd

from params import *

## Load data: Load only 2 .csv files
def load_data(INJURY_DF,PLAYLIST_DF):
    injury_path = os.path.join(DIR_PATH,DATA_FILE,INJURY_DF)
    playlist_path = os.path.join(DIR_PATH,DATA_FILE,PLAYLIST_DF)
    tracks_path = os.path.join(DIR_PATH,DATA_FILE,TRACKS_DF)
    injury_data = pd.read_csv(injury_path)
    playlist_data = pd.read_csv(playlist_path)

    print("===== Loaded data: injury_data and playlist_data  ======")
    return injury_data, playlist_data

## Data cleaning
def clean_injury_data(injury_data, playlist_data):
    # Step 1: Fetch games where injury PlayKey was not identified (NaN)
    GameID_nan = injury_data[injury_data.isna().any(axis = 1)].GameID

    # Step 2: Assume that injuries happened during the last PlayKey for all players where PlayKey is NaN
    # Fetch entries from playlist_data
    fetch_PlayKey = []

    for game in GameID_nan:
        fetch_PlayKey.append(playlist_data[playlist_data["GameID"] == game].PlayKey.iloc[-1])

    # Step 3: Get the indices of the NaN values in PlayKey
    nan_indices = injury_data[injury_data['PlayKey'].isna()].index

    # Step 4: Insert fetch_PlayKey values into injury_data at the NaN indices
    injury_data.loc[nan_indices, 'PlayKey'] = fetch_PlayKey

    # Step 5: Add a column Injured (1 = injured)
    injury_data['Injured'] = pd.Series([1 for x in range(len(injury_data.index))])

    # Step 6: Add a column to Injury_Class to injury_data representing the gravity of the injury
    # (1 = DM_M1, 2 = DM_M7, 3 = DM_M28, 4 = DM_M42)
    injury_data['Injury_Class'] = injury_data.DM_M1 + injury_data.DM_M7 + injury_data.DM_M28 + injury_data.DM_M42

    print("===== clean_injury_data ======")
    print(injury_data.head(1))

    return injury_data

def clean_playlist_data(playlist_data):
    # Temperature: Replace -999 with the 60F
    playlist_data["Temperature"].replace(-999, 60, inplace = True)

    ## Weather: Reduce Weather to 6 categories in total
    # Creat function
    def categorize_weather(description):
        if pd.isna(description) or "indoor" in description.lower() or "controlled" in description.lower():
            return "Indoor"
        elif "rain" in description.lower() or "shower" in description.lower() or "precip" in description.lower() or "snow" in description.lower():
            return "Precipitation"
        elif "cold" in description.lower() or "cool" in description.lower():
            return "Cold Weather"
        elif "hot" in description.lower() or "warm" in description.lower() or "heat" in description.lower():
            return "Extreme Heat"
        elif "cloud" in description.lower() or "overcast" in description.lower() or "hazy" in description.lower():
            return "Cloudy/Overcast"
        else:
            return "Clear/Normal Weather"

    # Apply the function to categorize weather
    playlist_data["WeatherGroup"] = playlist_data["Weather"].apply(categorize_weather)

    ## StadiumType: Reduce Stadium type to 3 categories
    # Unknown values are treated as outdoor since the latter is the most frequent.
    stadium_mapping = {"outdoor": "Outdoor",
                        "open air": "Outdoor",
                        "dome": "Indoor",
                        "indoor": "Indoor",
                        "retractable": "Hybrid",
                        "hybrid": "Hybrid",
                        "partial": "Hybrid",
                        "n/a": "Outdoor",
                        "unknown": "Outdoor",
                        "nan": "Outdoor",
                        "": "Outdoor"
                        }
    def map_category(value, mapping):
        value = str(value).lower()
        for key, category in mapping.items():
            if key in value:
                return category
        return "Other"
    # Apply the mappings to StadiumType
    playlist_data["StadiumTypeGroup"] = playlist_data["StadiumType"].apply(lambda x: map_category(x, stadium_mapping))

    ## PlayType: Fillna with the most frequent play type ("Pass")
    playlist_data["PlayType"].fillna("Pass", inplace = True)
    playlist_data["PlayType"].replace([0, '0', 0.0], "Pass", inplace = True)

    ## Position: Reduce categories to 3 (PositionCategory column) and then to 7 (PositionGranularCategory)
    # PositionCategory: Defensive, Offensive, Special Teams
    # Replace Missing Data by P as a poistion, this is a minority group
    playlist_data["Position"] = playlist_data["Position"].replace("Missing Data", "P")

    # Define the mapping for 3 categories
    position_category_map = {"QB": "Offensive",
                            "WR": "Offensive",
                            "RB": "Offensive",
                            "TE": "Offensive",
                            "G": "Offensive",
                            "T": "Offensive",
                            "C": "Offensive",
                            "HB": "Offensive",
                            "ILB": "Defensive",
                            "DE": "Defensive",
                            "FS": "Defensive",
                            "CB": "Defensive",
                            "OLB": "Defensive",
                            "DT": "Defensive",
                            "SS": "Defensive",
                            "MLB": "Defensive",
                            "NT": "Defensive",
                            "DB": "Defensive",
                            "LB": "Defensive",
                            "S": "Defensive",
                            "P": "Special Teams",
                            "K": "Special Teams"
                            }

    # Apply the mapping to create the PositionCategory column
    playlist_data["PositionCategory"] = playlist_data["Position"].map(position_category_map)

    # PositionGranularCategory:
    position_granular_map = {
                # Offensive Categories
                "QB": "Backfield",
                "RB": "Backfield",
                "HB": "Backfield",
                "WR": "Receivers",
                "TE": "Receivers",
                "G": "Offensive Line",
                "T": "Offensive Line",
                "C": "Offensive Line",

                # Defensive Categories
                "DE": "Defensive Line",
                "DT": "Defensive Line",
                "NT": "Defensive Line",
                "ILB": "Linebackers",
                "OLB": "Linebackers",
                "MLB": "Linebackers",
                "LB": "Linebackers",
                "CB": "Defensive Backs",
                "FS": "Defensive Backs",
                "SS": "Defensive Backs",
                "DB": "Defensive Backs",
                "S": "Defensive Backs",

                # Special Teams
                "P": "Kicking Unit",
                "K": "Kicking Unit"
                            }

# Apply the mapping to create the GranularPositionCategory column
    playlist_data["PositionGranularCategory"] = playlist_data["Position"].map(position_granular_map)

    print("===== clean_playlist_data ======")
    print(playlist_data.head(1))

    return playlist_data

## Feature Engineering
# Fatigue Feature
def engineered_fatigue(playlist_data):
    # Step 1: Sort data
    playlist_data = playlist_data.sort_values(["PlayerKey", "PlayerDay"])

    # Step 2: Calculate games played in the last 2 weeks (14 days)
    def games_in_two_weeks(player_days):
        return sum((player_days.max() - player_days) <= 14)

    playlist_data["RecentGames"] = playlist_data.groupby("PlayerKey")["PlayerDay"].transform(games_in_two_weeks)

    # Step 3: Calculate a cumulative games played, accounting for the total workload
    playlist_data["CumulativeGames"] = playlist_data.groupby("PlayerKey").cumcount() + 1

    # Step 4: Calculate days since last game, accounting for recovery time
    playlist_data["DaysSinceLastGame"] = playlist_data.groupby("PlayerKey")["PlayerDay"].diff()
    playlist_data["DaysSinceLastGame"] = playlist_data["DaysSinceLastGame"].fillna(0)  # First game has 0 days since last game

    # Step 5: Compute season progress
    playlist_data["SeasonProgress"] = playlist_data.groupby("PlayerKey").apply(
        lambda x: (x["PlayerDay"] - x["PlayerDay"].min()) / (x["PlayerDay"].max() - x["PlayerDay"].min()) * 100).reset_index(level=0, drop=True)

    # Final step: Calculate a fatigue score
    playlist_data["FatigueScore"] = (
        (playlist_data["RecentGames"] * 2) +  # Recent games have a higher impact
        (playlist_data["CumulativeGames"] * 0.5) +  # Cumulative games have a moderate impact
        (playlist_data["SeasonProgress"] * 0.1) -  # Season progress has a small impact
        (playlist_data["DaysSinceLastGame"] * 0.5)  # More days since last game reduces fatigue
    )

    # Remove negative values from Fatigue score
    playlist_data["FatigueScore"] = playlist_data["FatigueScore"].clip(lower=0)

    print("===== Added fatigue features to playlist_data ======")
    print(playlist_data.head(1))

    return playlist_data

### Merge Injury data and Playlist data
def merge_df(injury_data: pd.DataFrame,playlist_data: pd.DataFrame):
    mergePlayerKey = pd.merge(playlist_data, injury_data[["PlayerKey","Injured","Injury_Class"]], on = "PlayerKey", how = "left").fillna(0)
    mergeGameID = pd.merge(playlist_data, injury_data[["GameID","Injured","Injury_Class"]], on = "GameID", how = "left").fillna(0)
    mergePlayKey = pd.merge(playlist_data, injury_data[["PlayKey","Injured","Injury_Class"]], on = "PlayKey", how = "left").fillna(0)

    print("===== Merged DataFrame: Output 3 DataFrames ======")

    return mergePlayerKey, mergeGameID, mergePlayKey
