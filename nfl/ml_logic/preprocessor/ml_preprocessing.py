import pandas as pd

#### Load data
def load_data(INJURY_DF,PLAYLIST_DF,TRACKS_DF):
    injury_path = os.path.join(DIR_PATH,DATA_FILE,INJURY_DF)
    playlist_path = os.path.join(DIR_PATH,DATA_FILE,PLAYLIST_DF)
    injury_data = pd.read_csv(injury_path)
    playlist_data = pd.read_csv(playlist_path)

    return injury_data, playlist_data

#### Data cleaning and reduced categorization
### Create a function to cleanr injury data
def clean_injury_data(injury_data, playlist_data):
    # Fetch games where injury PlayKey was not identified (NaN)
    GameID_nan = injury_data[injury_data.isna().any(axis = 1)].GameID

    # Assume that injuries happened during the last PlayKey for all players where PlayKey is NaN.
    # Fetch information from playlist_data
    fetch_PlayKey = []

    for game in GameID_nan:
        fetch_PlayKey.append(playlist_data[playlist_data["GameID"] == game].PlayKey.iloc[-1])

    # Get the indices of the NaN values in PlayKey
    nan_indices = injury_data[injury_data['PlayKey'].isna()].index

    # Insert fetch_PlayKey values into injury_data at the NaN indices
    injury_data.loc[nan_indices, 'PlayKey'] = fetch_PlayKey

    # Add a column Injured: (1 = injured)
    injury_data['Injured'] = pd.Series([1 for x in range(len(injury_data.index))])

    # Add a column to Injury_Class to injury_data representing the gravity of the injury:
    # (1 = DM_M1, 2 = DM_M7, 3 = DM_M28, 4 = DM_M42)
    injury_data['Injury_Class'] = injury_data.DM_M1 + injury_data.DM_M7 + injury_data.DM_M28 + injury_data.DM_M42

    print("===== Clean injury_data ======")
    print(injury_data.head(1))

    return injury_data

### Create a function to clean playlist data
def clean_palylist_data(playlist_data):
    ## Temperature: Replace -999 with the 60F
    playlist_data["Temperature"].replace(-999, 60, inplace = True)

    ## Weather: Reduce weather catergories to 5
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

    ## StadiumType: Reduce stadium type to 3
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

    # Apply the mappings to column StadiumType
    playlist_data["StadiumTypeGroup"] = playlist_data["StadiumType"].apply(lambda x: map_category(x, stadium_mapping))

    ## Playtype: Replace NaN and 0 w/ "Pass" as a playtype
    playlist_data["PlayType"].fillna("Pass", inplace = True)
    playlist_data["PlayType"].replace([0, '0', 0.0], "Pass", inplace = True)

    print("===== Clean playlist_data ======")
    print(playlist_data.head(1))

    return playlist_data

### Feature Engineering
