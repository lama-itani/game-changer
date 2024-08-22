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

### Crossvalidation, Training, Model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

### Evaluation
from sklearn.metrics import confusion_matrix
from sklearn.metrics  import ConfusionMatrixDisplay

## Import from local packages
from preprocessor import ml_preprocessing

def ml_pipeline(clean_data: pd.DataFrame, mdl_type: str):
    # Change column names
    clean_data.rename(columns = {"DM_TOT": "Injury_Class"}, inplace = True)

    # Drop columns
    clean_data.drop(columns = ["Weather_Temp", "PlayerKey", "GameID", "PlayKey", "Surface"], inplace = True)

    # Preprocess
    num_transformer = make_pipeline(SimpleImputer(), RobustScaler())
    cat_transformer = OneHotEncoder()

    preproc = make_column_transformer(num_transformer, make_column_selector(dtype_include = ["float64","int64"]),
                                        cat_transformer, make_column_selector(dtype_include = "object"),
                                        remainder = "passthrough")
    # Add estimator
    if mdl_type == "LogisticRegression":
        mdl = LogisticRegression()
    elif mdl_type == "MultiOutputClassifier":
        mdl = MultiOutputClassifier(LogisticRegression())
    else:
        raise ValueError("Invalid model type")

    ml_pipe = make_pipeline(preproc, mdl)

    # Split data based on mdl_type: X remains unchanged regardless, y is impacted by mdl_type
    X = clean_data.drop(columns = ["Injured","Injury_Class"])
    if mdl_type == "LogisticRegression":
        y = clean_data.Injured
    elif mdl_type == "MultiOutputClassifier":
        y = clean_data.Injury_Class
    else:
        raise ValueError("Invalid model type")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    return ml_pipe, X_train, y_train, X_test, y_test
