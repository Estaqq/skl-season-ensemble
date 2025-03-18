"""
This is an estimator which manages an independent collection of base estimators to 
be fitted to data from different seasons.
"""

# Authors: Max Gläser
# License: BSD 3 clause

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, _fit_context
from sklearn.metrics import euclidean_distances
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import LogisticRegression
import math


class TemplateEstimator(BaseEstimator):
    """A template estimator to be used as a reference implementation. TODO

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters TODO
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstration of how to pass and store parameters.

    Attributes TODO
    ----------
    is_fitted_ : bool
        A boolean indicating whether the estimator has been fitted.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    Examples TODO
    --------
    >>> from skl_seas_ens import TemplateEstimator
    >>> import numpy as np
    >>> X = np.arange(100).reshape(100, 1)
    >>> y = np.zeros((100, ))
    >>> estimator = TemplateEstimator()
    >>> estimator.fit(X, y)
    TemplateEstimator()
    """

    # This is a dictionary allowing to define the type of parameters.
    # It used to validate parameter within the `_fit_context` decorator. TODO
    _parameter_constraints = {
        "demo_param": [str],
    }

    def __init__(self, demo_param="demo_param"):
        self.demo_param = demo_param

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """A reference implementation of a fitting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

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
        X, y = self._validate_data(X, y, accept_sparse=True)
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def predict(self, X):
        """A reference implementation of a predicting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        # Check if fit had been called
        check_is_fitted(self)
        # We need to set reset=False because we don't want to overwrite `n_features_in_`
        # `feature_names_in_` but only check that the shape is consistent.
        X = self._validate_data(X, accept_sparse=True, reset=False)
        return np.ones(X.shape[0], dtype=np.int64)


# Note that the mixin class should always be on the left of `BaseEstimator` to ensure
# the MRO works as expected.
class SeasonalClassifier(ClassifierMixin, BaseEstimator):
    """An classifier, which manages a collection of base estimators of type base_model_class and trains 
    and calls them depending on the value of the value of the feature of the name passed in time_column_name.

    Parameters TODO
    ----------
    demo_param : str, default='demo'
        A parameter used for demonstation of how to pass and store paramters.

    Attributes TODO
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
        "demo_param": [str],
    }

    def __init__(self, demo_param="demo"):
        self.demo_param = demo_param

    def __init__(
            self, 
            base_model_class = LogisticRegression,
            base_model_args = {}, 
            window_size = None,
            window_start = 0,
            window_end = 365,
            n_windows = 10,
            windows = [],
            padding = 105,
            time_column_name = 'day' ,
            drop_time_column = True,
            data_is_periodic = True
            ):
        self.base_model_class = base_model_class
        self.base_model_args = base_model_args
        self.window_size = window_size
        self.window_start = window_start
        self.window_end = window_end
        self.n_windows = n_windows
        self.windows = windows
        self.padding = padding
        self.time_column_name = time_column_name
        self.drop_time_column = drop_time_column   
        self.data_is_periodic = data_is_periodic
    

    def _set_up_windows(self):
        if len(self.windows > 0):
            self._internal_windows = self.windows
        elif self.window_size is not None:
            self._internal_windows = np.arange(self.window_start, self.window_end, self.window_size)
        else:
            size = math.ceil((self.window_end - self.window_start) / self.n_windows)
            self.windows = np.arange(self.window_start, self.window_end, size)

    def _get_window(self, day):
        for i in range(len(self.windows) - 1):
            if self._internal_windows[0] <= day < self._internal_windows[1]:
                return i
        raise ValueError("Value of " + self.time_column_name + "is out of bounds.")
    
    def _create_models(self):
        self._models = []
        for i in range( len(self.windows) -1 ):
            self._models.append(self.base_model_class(**self.base_model_args))

    def _select_rows(self, data, window_index):
        start = self._internal_windows[window_index] - self.padding
        end = self._internal_windows[window_index+1] + self.padding
        selection = data[(data[self.time_column_name] >= start) & (data[self.time_column_name] < end)]
        if self.data_is_periodic:
            if start < self.window_start:
                selection = pd.concat(selection, data[data[self.time_column_name] >= self.window_end - (self.window_start - start)])
            if end >= self.window_end:
                selection = pd.concat(selection, data[data[self.time_column_name] < self.window_start + (self.window_end - end)])
        return selection
    
    def _fit_base_models(self):
        for i in range(len(self._models)):
            selection = self._select_rows(self.X_, i)
            if(self.drop_time_column):
                self.models[i].fit(selection.drop(columns=[self.time_column_name]), self.y_[selection])
            else:
                self.models[i].fit(self.X_[selection], self.y_[selection])

    
    def _apply_appropriate_model(self, row):
        window = self._get_window(row[self.time_column_name])
        #X = pd.DataFrame(X)
        #X = X.reindex(columns=self.feature_names_in_)
        #features = features.drop('id', axis=1)
        if self.drop_time_column:
            row = row.drop(self.time_column_name, axis=1)
        model = self.models[window]
        return model.predict(row)[0]
        



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
        X, y = self._validate_data(X, y)
        # We need to make sure that we have a classification task
        check_classification_targets(y)

        # classifier should always store the classes seen during `fit`
        self.classes_ = np.unique(y)

        # Store the training data to predict later
        self.X_ = X
        self.y_ = y

        self._set_up_windows()
        self._create_models()
        self._fit_base_models()
        self.is_fitted_ = True

        # Return the classifier
        return self

    def predict(self, X):
        """A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        # Check if fit had been called
        check_is_fitted(self)

        # Input validation
        # We need to set reset=False because we don't want to overwrite `n_features_in_`
        # `feature_names_in_` but only check that the shape is consistent.
        X = self._validate_data(X, reset=False)

        prediction = X.apply(self._apply_appropriate_model, axis='columns')
        return prediction
