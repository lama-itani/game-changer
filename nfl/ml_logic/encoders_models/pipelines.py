"""
This file takes in clean data from ml_preprocessing.py and preprocess (scaling & encodine).
"""

### Import libraries ###
# 1 - DATA MANIPULATION
import pandas as pd
import numpy as np

# 2 - MACHINE LEARNING
### Preprocess
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import make_union
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
from sklearn.impute import SimpleImputer

### Scalers
from sklearn.preprocessing import RobustScaler

### Encoders
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder

### Crossvalidation, Training, Model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import cross_val_score


### Evaluation
from sklearn.metrics import confusion_matrix
from sklearn.metrics  import ConfusionMatrixDisplay

## Import from local packages
from preprocessor import dl_preprocessing

def dl_pipeline(cleaned_df):

    # Preprocess
    num_transformer = RobustScaler()
    cat_transformer = OneHotEncoder(sparse_output = False, drop="if_binary")
    ordinal_transformer = OrdinalEncoder()

    num_col = ['time',"x",'true_dist', 'turn', 'turn_agg', 'true_speed', 'dir_o_diff', '45_turn',
       '180_turn', 'cumsum_45', 'cumsum_180', 'PlayerDay',
       'Temperature', 'PlayerGamePlay', 'max_game', 'playkey_max']
    cat_col = ['event', 'RosterPosition', 'StadiumType', 'FieldType', 'Weather',
       'PlayType', 'Position','Wet', 'Indoor','BodyPart']
    ordinal_col =['Temperature_transformed']

    preproc = make_column_transformer(
        (num_transformer, num_col),
        (cat_transformer, cat_col),
        (ordinal_transformer,ordinal_col),
        remainder = "drop")

    dl_pipe = make_pipeline(preproc, MultiOutputClassifier(LogisticRegression()))

    ## Split data
    X = cleaned_df.drop(columns = ["injury_duration"])
    # y is a dataframe with two columns to be encoded prior to training mdl
    y = cleaned_df[["injury_duration"]]

    ohe_targer = OneHotEncoder(sparse_output = False, drop="if_binary")
    encoded_y = ohe_targer.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, encoded_y, test_size=0.2)
    print("====== Pipeline done ======")
    return dl_pipe, X_train, y_train, X_test, y_test

def dl_cross_val(dl_pipe, X_train, y_train, X_test, y_test):
    print("====== CROSSVAL started ======")
    result = cross_val_score(dl_pipe, X_train, y_train, cv=5, scoring='r2').mean()
    print(result)
    print("====== CROSSVAL started ======")
    return result
