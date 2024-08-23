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
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score, precision_score

### Evaluation
from sklearn.metrics import confusion_matrix
from sklearn.metrics  import ConfusionMatrixDisplay

### Save model
from joblib import dump

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
        mdl = LogisticRegression(max_iter = 10_000)
    elif mdl_type == "MultiOutputClassifier":
        mdl = MultiOutputClassifier(LogisticRegression())
    else:
        raise ValueError("Invalid model type")

    ml_pipe = make_pipeline(preproc, mdl)

    ## Split data
    X = clean_data.drop(columns = ["Injured","Injury_Class"])
    if mdl_type == "LogisticRegression":
        y = clean_data.Injured
    elif mdl_type == "MultiOutputClassifier":
        y = clean_data.Injury_Class
        ohe = OneHotEncoder(handle_unknown = "ignore", sparse_output = False)
        y = pd.DataFrame(ohe.fit_transform(np.array(y).reshape(-1,1)))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    return ml_pipe, mdl_type, X_train, y_train, X_test, y_test

def ml_train(ml_pipe, mdl_type, X_train, y_train, X_test, y_test):
    #Train pipeline
    ml_pipe.fit(X_train, y_train)

    # Make predictions
    y_pred = ml_pipe.predict(X_test)

    # Score model
    if mdl_type == "LogisticRegression":
        precision = precision_score(y_test, y_pred, average="weighted")
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Precision = {precision}, Accuracy = {accuracy}")
    elif mdl_type == "MultiOutputClaissifier":
        score = [precision_score(y_test, y_pred, average='samples'), accuracy_score(y_test, y_pred)]
        print (f"Precision = {precision}, Accuracy = {accuracy}")

    # Save the model
    model_filename = f"trained_{mdl_type}.joblib"
    dump(ml_pipe, model_filename)
    print(f"Model saved as {model_filename}")

    return y_pred, precision, accuracy