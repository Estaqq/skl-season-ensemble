import pytest
from sklearn.base import clone
from skl_seas_ens import SeasonalClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate

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
    X, y = make_classification(n_samples=20, n_features=8, random_state=42)
    # Generate random time data between 1 and 7
    
    # Create a DataFrame
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    time_data = pd.Series(np.random.randint(1, 8, size=len(y)))
    df['time'] = time_data
    
    # Initialize the SeasonalClassifier with RandomForestClassifier
    seasonal_clf = SeasonalClassifier(base_model_class=RandomForestClassifier,time_column= 'time', n_windows=1, base_model_args={'random_state': 42}, col_names= df.columns)
    
    # Fit the classifier
    seasonal_clf.fit(df, y)
    
    # Predict
    predictions = seasonal_clf.predict(df)
    
    # Check the length of predictions
    assert len(predictions) == len(df)

def test_seasonal_classifier_with_cross_validate():
    # Create a random dataset
    X, y = make_classification(n_samples=100, n_features=8, random_state=42)
    
    # Initialize the SeasonalClassifier with RandomForestClassifier
    seasonal_clf = SeasonalClassifier(base_model_class=RandomForestClassifier, n_windows=5, base_model_args={'random_state': 42})
    
    # Perform cross-validation
    cv_results = cross_validate(seasonal_clf, X, y, cv=3)
    
    # Check that cross-validation results contain the expected keys
    assert 'test_score' in cv_results
    assert 'fit_time' in cv_results
    assert 'score_time' in cv_results
    
    # Check that the cross-validation results have 5 entries (one for each fold)
    assert len(cv_results['test_score']) == 3
    assert len(cv_results['fit_time']) == 3
    assert len(cv_results['score_time']) == 3

def test_cross_validate_same_as_base():
    for baseclass in [RandomForestClassifier, LogisticRegression]:
        # Create a random dataset
        X, y = make_classification(n_samples=100, n_features=8, random_state=42)
        
        # Initialize the base classifier
        base_clf = baseclass(random_state=42)
        
        # Initialize the SeasonalClassifier with the base classifier
        seasonal_clf = SeasonalClassifier(base_model_class=baseclass, n_windows=1, base_model_args={'random_state': 42})
        
        # Perform cross-validation with the base classifier
        base_cv_results = cross_validate(base_clf, X, y, cv=3)
        
        # Perform cross-validation with the SeasonalClassifier
        seasonal_cv_results = cross_validate(seasonal_clf, X, y, cv=3)
        
        # Check that the cross-validation scores are the same
        assert np.allclose(base_cv_results['test_score'], seasonal_cv_results['test_score'])

def test_cross_validate_with_drop_time():
    for baseclass in [RandomForestClassifier, LogisticRegression]:
        # Create a random dataset
        X, y = make_classification(n_samples=100, n_features=8, random_state=42)
        
        # Generate random time data between 1 and 7
        time_data = np.random.randint(1, 8, size=len(y))
        
        # Create a DataFrame
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        df['time'] = time_data
        
        # Initialize the SeasonalClassifier with the base classifier and drop_time=True
        seasonal_clf = SeasonalClassifier(base_model_class=baseclass, time_column='time', n_windows=1, base_model_args={'random_state': 42}, drop_time_column=True, col_names=df.columns)
        
        # Perform cross-validation with the SeasonalClassifier
        seasonal_cv_results = cross_validate(seasonal_clf, df, y, cv=3)
        
        # Initialize the base classifier
        base_clf = baseclass(random_state=42)
        
        # Perform cross-validation with the base classifier
        base_cv_results = cross_validate(base_clf, df.drop(columns=['time']), y, cv=3)
        
        # Check that the cross-validation scores are the same
        assert np.allclose(base_cv_results['test_score'], seasonal_cv_results['test_score'])

        
def test_cross_validate_same_as_base_ROC_ACC():
    scoring = {'AUC': 'roc_auc', 'Accuracy': 'accuracy'}
    for baseclass in [RandomForestClassifier, LogisticRegression]:
        # Create a random dataset
        X, y = make_classification(n_samples=100, n_features=8, random_state=42)
        
        # Initialize the base classifier
        base_clf = baseclass(random_state=42)
        
        # Initialize the SeasonalClassifier with the base classifier
        seasonal_clf = SeasonalClassifier(base_model_class=baseclass, n_windows=1, base_model_args={'random_state': 42})
        
        # Perform cross-validation with the base classifier
        base_cv_results = cross_validate(base_clf, X, y, cv=3, scoring=scoring)
        
        # Perform cross-validation with the SeasonalClassifier
        seasonal_cv_results = cross_validate(seasonal_clf, X, y, cv=3, scoring=scoring)
        
        # Check that the cross-validation scores are the same
        for metric in scoring.keys():
            assert np.allclose(base_cv_results[f'test_{metric}'], seasonal_cv_results[f'test_{metric}'])

def test_data_is_periodic_irrelevant_with_one_window():
    # Create a random dataset
    X, y = make_classification(n_samples=100, n_features=8, random_state=42)
    
    # Initialize the SeasonalClassifier with data_is_periodic=True and n_windows=1
    seasonal_clf_periodic = SeasonalClassifier(base_model_class=LogisticRegression, n_windows=1, base_model_args={'random_state': 42}, data_is_periodic=True)
    
    # Initialize the SeasonalClassifier with data_is_periodic=False and n_windows=1
    seasonal_clf_non_periodic = SeasonalClassifier(base_model_class=LogisticRegression, n_windows=1, base_model_args={'random_state': 42}, data_is_periodic=False)
    
    # Fit both classifiers
    seasonal_clf_periodic.fit(X, y)
    seasonal_clf_non_periodic.fit(X, y)
    
    # Predict with both classifiers
    periodic_pred = seasonal_clf_periodic.predict(X)
    non_periodic_pred = seasonal_clf_non_periodic.predict(X)
    
    # Assert that the predictions are the same
    assert (periodic_pred == non_periodic_pred).all()

def test_padding_irrelevant_with_one_window():
    # Create a random dataset
    X, y = make_classification(n_samples=100, n_features=8, random_state=42)
    
    # Initialize the SeasonalClassifier with padding=0, n_windows=1, and data_is_periodic=True
    seasonal_clf_no_padding = SeasonalClassifier(base_model_class=LogisticRegression, n_windows=1, base_model_args={'random_state': 42}, padding=0, data_is_periodic=True)
    
    # Initialize the SeasonalClassifier with padding=100, n_windows=1, and data_is_periodic=True
    seasonal_clf_with_padding = SeasonalClassifier(base_model_class=LogisticRegression, n_windows=1, base_model_args={'random_state': 42}, padding=100, data_is_periodic=True)
    
    # Fit both classifiers
    seasonal_clf_no_padding.fit(X, y)
    seasonal_clf_with_padding.fit(X, y)
    
    # Predict with both classifiers
    no_padding_pred = seasonal_clf_no_padding.predict(X)
    with_padding_pred = seasonal_clf_with_padding.predict(X)
    
    # Assert that the predictions are the same
    assert (no_padding_pred == with_padding_pred).all()


def test_select_rows_with_one_window():
    # Create a random dataset
    X, y = make_classification(n_samples=100, n_features=8, random_state=0)
    
    # Generate random time data between 1 and 7
    time_data = np.random.randint(1, 8, size=len(y))
    
    # Create a DataFrame
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['target'] = y
    df['time'] = time_data
    X= df.drop(columns=['target'])

    # Initialize the SeasonalClassifier with one window
    seasonal_clf = SeasonalClassifier(base_model_class=LogisticRegression, time_column='time', n_windows=1, base_model_args={'random_state': 42}, col_names=X.columns)    
    # Fit the classifier
    seasonal_clf.fit(X, df['target'])
    
    assert seasonal_clf._time_column == X.columns.get_loc('time')

    # Select rows for the single window
    selected_rows = seasonal_clf._select_rows(df.values, 0)
    
    # Assert that all rows are selected
    assert selected_rows.all()
    assert len(selected_rows) == len(df)

def test_select_rows_with_multiple_windows():
    X, y = make_classification(n_samples=100, n_features=8, random_state=0)
    
    # Generate random time data between 1 and 7
    time_data = np.random.randint(1, 8, size=len(y))

    # Create a DataFrame
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['target'] = y
    df['time'] = time_data
    X= df.drop(columns=['target'])

    seasonal_clf = SeasonalClassifier(base_model_class=LogisticRegression, time_column='time', n_windows=10, 
                                      base_model_args={'random_state': 42}, col_names=X.columns,padding = 400)    
    # Fit the classifier
    seasonal_clf.fit(X, df['target'])
    
    # Select rows for the single window
    selected_rows = seasonal_clf._select_rows(df.values, 1)
    
    # Assert that all rows are selected
    assert selected_rows.all()
    assert len(selected_rows) == len(df)

def test_fit_base_models():
    # Load the dataset from the CSV file
    df = pd.read_csv('examples/data/train.csv')
    
    # Extract features and target
    X = df.drop(columns=['rainfall'])
    y = df['rainfall']
    
    # Initialize and fit SeasonalClassifier with one window
    seasonal_clf = SeasonalClassifier(base_model_class=LogisticRegression, base_model_args={'random_state': 0, 'max_iter': 10000},
                                       time_column='day', n_windows=1,
                                         col_names=df.columns)
    seasonal_clf.fit(X, y)

    # Copy and refit the first base model
    copied_model = clone(seasonal_clf._models[0])
    copied_model.fit(X, y)

    # Assert that the coefficients are the same
    assert (copied_model.coef_ == seasonal_clf._models[0].coef_).all()

    copied_model.fit(X[seasonal_clf._select_rows(X.values, 0)], y)
    assert (copied_model.coef_ == seasonal_clf._models[0].coef_).all()

    base_model = LogisticRegression(random_state=0,max_iter=10000)
    base_model.fit(X,y)
    assert (base_model.coef_ == copied_model.coef_).all()


def test_seasonal_classifier_drop_time_data():
    # Load the dataset from the CSV file
    df = pd.read_csv('examples/data/train.csv')
    # Extract features and target
    X = df.drop(columns=['rainfall'])
    #X = df.drop(columns=['rainfall','id'])

    y = df['rainfall']
    
    # Initialize the base classifier
    base_clf = LogisticRegression(random_state=0,max_iter=10000)
    
    # Initialize the SeasonalClassifier with the base classifier
    seasonal_clf = SeasonalClassifier(base_model_class=LogisticRegression, n_windows=15, padding= 20,
                                      base_model_args={'random_state': 0,'max_iter' : 10000}, 
                                      time_column='day', data_is_periodic=True, 
                                      drop_time_column=True,
                                      col_names= df.columns)
    
    # Fit both classifiers
    X_drop_time = X.drop(columns=['day']).values
    base_clf.fit(X_drop_time, y)
    seasonal_clf.fit(X, y)
    
    assert (X_drop_time == np.delete(seasonal_clf.X_, seasonal_clf._time_column, axis = 1)).all()
    


def test_seasonal_classifier_fit_base_model_drop_data():
    # Load the dataset from the CSV file
    df = pd.read_csv('examples/data/train.csv')
    
    # Extract features and target
    X = df.drop(columns=['rainfall'])
    #X = df.drop(columns=['rainfall','id'])

    X['dummy'] = df['id']
    y = df['rainfall']
    
    # Initialize the base classifier
    base_clf = LogisticRegression(random_state=0,max_iter=10000)
    
    # Initialize the SeasonalClassifier with the base classifier
    seasonal_clf = SeasonalClassifier(base_model_class=LogisticRegression, n_windows=1, 
                                      base_model_args={'random_state': 0,'max_iter' : 10000}, 
                                      time_column='day', data_is_periodic=True, 
                                      drop_time_column=True,
                                      col_names= df.columns)
    
    # Fit both classifiers
    X_drop_time = X.drop(columns=['day']).values
    base_clf.fit(X_drop_time, y)
    seasonal_clf.fit(X, y)
    
    assert (X_drop_time == np.delete(seasonal_clf.X_, seasonal_clf._time_column, axis = 1)).all()

    #base_clf.fit(X.drop(columns=['day']), y)
    
    # Predict with both classifiers
    #base_pred = base_clf.predict(X.drop(columns=['day']))
    base_pred = base_clf.predict(X_drop_time)

    seasonal_pred = seasonal_clf.predict(X)
    
    assert (base_clf.coef_ == seasonal_clf._models[0].coef_).all()

    # Assert that the predictions are the same
    assert (base_pred == seasonal_pred).all()
