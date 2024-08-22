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

def ml_pipeline(clean_data, mdl_type):
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

    return ml_pipe
