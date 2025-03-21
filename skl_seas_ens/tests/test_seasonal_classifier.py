import pytest
from sklearn.base import clone
from skl_seas_ens import SeasonalClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

def _test_against_baseclass(baseclass):
    # Create a random dataset
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    
    # Initialize the base classifier
    base_clf = baseclass(random_state=42)
    
    # Initialize the SeasonalClassifier with the base classifier
    seasonal_clf = SeasonalClassifier(base_model_class=baseclass,n_windows = 1, base_model_args={'random_state': 42})
    
    # Fit both classifiers
    base_clf.fit(X, y)
    seasonal_clf.fit(X, y)
    
    # Predict with both classifiers
    base_pred = base_clf.predict(X)
    seasonal_pred = seasonal_clf.predict(X)
    
    # Assert that the predictions are the same
    assert (base_pred == seasonal_pred).all()

def test_seasonal_classifier_same_as_base():
    for baseclass in [RandomForestClassifier, LogisticRegression]:
        _test_against_baseclass(baseclass)


def test_seasonal_classifier_with_dataframe():
    # Create a random dataset
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    # Generate random time data between 1 and 7
    
    # Create a DataFrame
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    time_data = pd.Series(np.random.randint(1, 8, size=len(y)))
    df['target'] = y
    df['time'] = time_data
    
    # Initialize the SeasonalClassifier with RandomForestClassifier
    seasonal_clf = SeasonalClassifier(base_model_class=RandomForestClassifier,time_column= 'time', n_windows=1, base_model_args={'random_state': 42}, col_names= df.columns)
    
    # Fit the classifier
    seasonal_clf.fit(df.drop(columns=['target']), df['target'])
    
    # Predict
    predictions = seasonal_clf.predict(df.drop(columns=['target']))
    
    # Check the length of predictions
    assert len(predictions) == len(df)