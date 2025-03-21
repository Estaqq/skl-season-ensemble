"""
This is an estimator which manages an independent collection of base estimators to 
be fitted to data from different seasons.
"""

# Authors: Max Gläser
# License: BSD 3 clause

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, check_X_y
from sklearn.linear_model import LogisticRegression
#import sklearn.utils.validation as validation
import math



# Note that the mixin class should always be on the left of `BaseEstimator` to ensure
# the MRO works as expected.
class SeasonalClassifier(ClassifierMixin, BaseEstimator):
    """An classifier, which manages a collection of base estimators of type base_model_class and trains 
    and calls them depending on the value of the feature of the name or index passed in time_column.

    Parameters 
    ----------

    base_model_class : BaseEstimator, default=LogisticRegression
        The class of the base model to be used for the classification.
    base_model_args : dict, default=None
        Arguments to be passed to the base model class.
    window_size : int, float, default=None
        The size of the windows into which the data is split.
    window_start : int, float, default=None
        The lower bound for the range for which we expect data. If None, the minimum value occuring in the training data is used.
    window_end : int, float, default=None
        The upper bound for the range for which we expect data. If None, the maximum value occuring in the training data is used.
    n_windows : int, default=10
        The number of windows in which the data is split. If window_size is set, this parameter is ignored.
    windows : list, default=None
        A list of the boundaries of the windows. If set, window_size, window_start and window_end are ignored.
    padding : int, float, default=105
        The distance in time units away from some window for which data is still considered for training the classifier for that particular window.
    time_column : str, int, default=0
        The index or name of the column in the input data that contains the time information based on which data is assigned to base classifiers.
        Strings can only be passed, if the input data is a pandas DataFrame or Series.
        When using SeasonalClassifier together with functions such as grid_search or cross_validate, either a numerical index must be provided,
        or the argument col_names must also be provided, since those functions pass numpy arrays during their invocation of SeasonalClassifier.
    col_names : list, default=None
        The names of the columns in the input data. Only needed when the time_column is a string.
    drop_time_column : bool, default=True
        Whether to drop the time column from the input data before passing it to the base classifiers. Ignored when the input data has only one feature.
    data_is_periodic : bool, default=True
        Whether the data is periodic. If True, data is considered to be periodic and the windows are wrapped around the window_start and window_end values.

    Attributes 
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.

    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.

    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    Examples
    --------
    TODO
    """

    # This is a dictionary allowing to define the type of parameters.
    # It used to validate parameter within the `_fit_context` decorator.
    _parameter_constraints = {
        "window_size": [int, float, None],
        "window_start": [int, float, None],
        "window_end": [int, float, None],
        "n_windows": [int, None],
        "windows": [list, None],
        "padding": [int, float, None],
        "time_column": [str, int, None],
        "drop_time_column": [bool, None],
        "data_is_periodic": [bool, None],
        "verbose": [bool]
    }

    def __init__(
            self, 
            base_model_class = LogisticRegression,
            base_model_args = None, 
            window_size: int | float | None = None,
            window_start: int | float | None = None,
            window_end: int | float | None = None,
            n_windows: int | None  = 10,
            windows: list | None = None,
            padding: int | float = 105,
            time_column: str | int = 0,
            col_names : list | None = None,
            drop_time_column: bool = False,
            data_is_periodic: bool = True,
            verbose: bool = False
            ):
        super().__init__()
        self.base_model_class = base_model_class
        self.base_model_args = base_model_args
        self.window_size = window_size
        self.window_start = window_start
        self.window_end = window_end
        self.n_windows = n_windows
        self.windows = windows
        self.padding = padding
        self.time_column = time_column
        self.drop_time_column = drop_time_column   
        self.data_is_periodic = data_is_periodic
        self.col_names = col_names
        self.verbose = verbose

    def _set_up_windows(self):
        if self.windows != None:
            self._internal_windows = self.windows
            assert len(self._internal_windows) > 1
        else:
            self._internal_windows = np.linspace(self._window_start, self._window_end, max(2, self._n_windows))

    def _handle_out_of_sample_time(self, day):
        if self.data_is_periodic:
            return self._window_start + (day - self._window_start) % (self._window_end - self._window_start)
        else:
            if day <= self._window_start:
                return self._window_start
            elif day >= self._window_end:
                return self._window_end 
            else:
                return day
        return

    def _get_window(self, day):
        day = self._handle_out_of_sample_time(day)
        if len(self._internal_windows == 1):
            return 0
        for i in range(len(self.windows) - 1):
            if self._internal_windows[i] <= day < self._internal_windows[i+1]:
                return i
        if day == self._internal_windows[len(self._internal_windows) - 1]:
            return len(self._internal_windows) - 2
        raise ValueError("Value of time column is out of bounds.")
    
    def _create_models(self):
        if self.base_model_args == None:
            self._base_model_args = {}
        else:
            self._base_model_args = self.base_model_args
        self._models = []
        assert len(self._internal_windows) > 1
        for i in range( len(self._internal_windows) -1 ):
            self._models.append(self.base_model_class(**self._base_model_args))

    def _select_rows(self, data, window_index):
        start = self._internal_windows[window_index] - self.padding
        end = self._internal_windows[window_index+1] + self.padding

        mask = (data[:,self._time_column] >= start) & (data[:,self._time_column] < end)
        if self.data_is_periodic:
            if start < self._window_start:
                mask1 = (data[:,self._time_column] >= self._window_end - (self._window_start - start))
                mask = mask | mask1
            if end >= self._window_end:
                mask2 = (data[:,self._time_column] < self._window_start + (end - self._window_end))
                mask = mask | mask2
        return mask
    
    def _fit_base_models(self):
        for i in range(len(self._models)):
            assert self.n_features_in_ == self.X_.shape[1]
            selection = self._select_rows(self.X_, i)
            if(self._drop_time_column):
                self._models[i].fit(np.delete(self.X_[selection,:], self._time_column,axis = 1), self.y_[selection])
            else:
                self._models[i].fit(self.X_[selection,:], self.y_[selection])
            if self.verbose:
                print(f"Model {i} fitted with {len(self.X_[selection])} samples.")  
    
    def _apply_appropriate_model(self, row, str_func = 'predict'):
        #Applies the function whose name is given in the string func of the appropriate model to a row of data.
        window = self._get_window(row[self._time_column])
        if self._drop_time_column:
            row = np.delete(row, self._time_column)
        model = self._models[window]
        row = row.reshape(1, -1)
        if not hasattr(model, str_func):
            raise AttributeError(f"The base model does not have the method '{str_func}'")
        func = getattr(model, str_func)
        return func(row)
        
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """A reference implementation of a fitting function for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """


        # `_validate_data` is defined in the `BaseEstimator` class.
        # It allows to:
        # - run different checks on the input data;
        # - define some attributes associated to the input data: `n_features_in_` and
        #   `feature_names_in_`.
        check_X_y(X, y)
        X, y = self._validate_data(X, y)
        
        #TODO: Why does new scikit 1.6 version of input validation not work?
        #X,y = validation.validate_data(X, y)

        self._validate_params()
        check_classification_targets(y)
        # classifier should always store the classes seen during `fit`
        self.classes_ = np.unique(y)




        # preprocess parameters
        if type(self.time_column) is str:
            if ((type(X) is pd.DataFrame) or (type(X) is pd.Series)):
                self._time_column = X.columns.get_loc(self.time_column) - 1
            elif self.col_names is not None:
                self._time_column = self.col_names.get_loc(self.time_column)
                assert (type(self._time_column) is int)
            else:
                raise ValueError("SeasonalClassifier If the time_column is provided as a string, the column names must be provided as well.")
        else: #time_column is an integer
            self._time_column = self.time_column

        # Store the training data to predict later
        if type(X) is pd.DataFrame or type(X) is pd.Series:
            self.X_ = X.values
        else:
            self.X_ = X
        if type(y) is pd.Series:
            self.y_ = y.values
        else:
            self.y_ = y
        assert self.n_features_in_ == self.X_.shape[1], "Number of columns in X_ does not match n_features_in_"

        if self.window_end == None:
            self._window_end = self.X_[:,self._time_column].max()
        else:
            self._window_end = self.window_end
        if self.window_start == None:
            self._window_start = self.X_[:,self._time_column].min()
        else:
            self._window_start = self.window_start
        if self.window_size == None:
            self._n_windows = self.n_windows
        else:
            self._n_windows = math.ceil((self._window_end - self._window_start) / self.window_size)
        if self.n_features_in_ == 1:
            self._drop_time_column = False
        else:
            self._drop_time_column = self.drop_time_column

        self._set_up_windows()
        self._create_models()
        self._fit_base_models()
        self.is_fitted_ = True

        # Return the classifier
        return self

    def predict(self, X):
        """ Predict using the appropriate base model. 

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The predicted labels for the input samples.
        """
        # Check if fit had been called
        check_is_fitted(self)

        # Input validation
        # We need to set reset=False because we don't want to overwrite `n_features_in_`
        # `feature_names_in_` but only check that the shape is consistent.
        X = self._validate_data(X, reset=False)

        #prediction = X.apply(self._apply_appropriate_model, axis='columns')
        prediction = np.apply_along_axis(self._apply_appropriate_model, 1, X).reshape(-1)
        return prediction

    def predict_proba(self, X):
        """Predict the probabilities of the input samples belonging to each class using the appropriate base model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        y : ndarray, shape (n_samples, n_classes)
            The predicted probabilities of the input samples belonging to each class.
        """
        check_is_fitted(self)
        X = self._validate_data(X, reset=False)

        apply_proba = lambda row: self._apply_appropriate_model(row, str_func='predict_proba')
        prediction = np.apply_along_axis(apply_proba, 1, X).reshape(-1, len(self.classes_))
        return prediction

    def predict_log_proba(self, X):
        """Predict the log-probabilities of the input samples belonging to each class using the appropriate base model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        y : ndarray, shape (n_samples, n_classes)
            The predicted log.probabilities of the input samples belonging to each class.

        """
        check_is_fitted(self)
        X = self._validate_data(X, reset=False)

        apply_log_proba = lambda row: self._apply_appropriate_model(row, str_func='predict_log_proba')
        prediction = np.apply_along_axis(apply_log_proba, 1, X).reshape(-1, len(self.classes_))
        return prediction