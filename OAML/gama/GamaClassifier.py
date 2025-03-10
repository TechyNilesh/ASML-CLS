import inspect
from typing import Union, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.base import ClassifierMixin
from river.base import Classifier


from .gama import Gama
from gama.data_loading import X_y_from_file
from gama.configuration.classification import clf_config
from gama.configuration.river_classification import clf_config_online
from gama.utilities.metrics import scoring_to_metric


class GamaClassifier(Gama):
    """Gama with adaptations for (multi-class) classification."""

    def __init__(
        self,
        config=None,
        scoring="neg_log_loss",
        online_learning=False,
        *args,
        **kwargs
    ) -> None:

        self._scoring = scoring

        if not config:
            # Do this to avoid the whole dictionary being included in the documentation.
            if not online_learning:
                config = clf_config
            else:
                config = clf_config_online

        self._metrics = scoring_to_metric(scoring)

        if any(metric.requires_probabilities for metric in self._metrics):
            # we don't want classifiers that do not have `predict_proba`,
            # because then we have to start doing one hot encodings of predictions etc.
            config = {
                alg: hp
                for (alg, hp) in config.items()
                if not (
                    inspect.isclass(alg)
                    and any(
                        issubclass(alg, baseclass)
                        for baseclass in [ClassifierMixin, Classifier]
                    )
                    and not any(
                        hasattr(alg(), attr)
                        for attr in ["predict_proba", "predict_proba_one"]
                    )
                )
            }

        self._label_encoder = None
        super().__init__(  # type: ignore
            *args,
            **kwargs,
            config=config,
            scoring=scoring,
            online_learning=online_learning
        )

    def _predict(self, x: pd.DataFrame):
        """Predict the target for input X.

        Parameters
        ----------
        x: pandas.DataFrame
            A dataframe with the same number of columns as the input to `fit`.

        Returns
        -------
        numpy.ndarray
            Array with predictions of shape (N,) where N is len(X).
        """
        if not self._online_learning:
            y = self.model.predict(x)  # type: ignore
            if self._label_encoder is not None:
                if y[0] not in self._label_encoder.classes_:
                    y = self._label_encoder.inverse_transform(y)
        else:
            """
            y_pred = []
            for x_i in x:
                y_pred.append(self.model.predict_one(x_i))
            y = np.array(y_pred)"""
            y = 999  # not implemented
        # Decode the predicted labels - necessary only if ensemble is not used.
        return y

    def _predict_proba(self, x: pd.DataFrame):
        """Predict the class probabilities for input x.

        Predict target for x, using the best found pipeline(s) during the `fit` call.

        Parameters
        ----------
        x: pandas.DataFrame
            A dataframe with the same number of columns as the input to `fit`.

        Returns
        -------
        numpy.ndarray
            Array of shape (N, K) with class probabilities where N is len(x),
             and K is the number of class labels found in `y` of `fit`.
        """
        return self.model.predict_proba(x)  # type: ignore

    def predict_proba(self, x: Union[pd.DataFrame, np.ndarray]):
        """Predict the class probabilities for input x.

        Predict target for x, using the best found pipeline(s) during the `fit` call.

        Parameters
        ----------
        x: pandas.DataFrame or numpy.ndarray
            Data with the same number of columns as the input to `fit`.

        Returns
        -------
        numpy.ndarray
            Array of shape (N, K) with class probabilities where N is len(x),
             and K is the number of class labels found in `y` of `fit`.
        """
        x = self._prepare_for_prediction(x)
        return self._predict_proba(x)

    def predict_proba_from_file(
        self,
        arff_file_path: str,
        target_column: Optional[str] = None,
        encoding: Optional[str] = None,
    ):
        """Predict the class probabilities for input in the arff_file.

        Parameters
        ----------
        arff_file_path: str
            An ARFF file with the same columns as the one that used in fit.
            Target column must be present in file, but its values are ignored.
        target_column: str, optional (default=None)
            Specifies which column the model should predict.
            If left None, the last column is taken to be the target.
        encoding: str, optional
            Encoding of the ARFF file.

        Returns
        -------
        numpy.ndarray
            Numpy array with class probabilities.
            The array is of shape (N, K) where N is len(X),
            and K is the number of class labels found in `y` of `fit`.
        """
        x, _ = X_y_from_file(arff_file_path, target_column, encoding)
        x = self._prepare_for_prediction(x)
        return self._predict_proba(x)

    def fit(self, x, y, *args, **kwargs):
        """Should use base class documentation."""
        y_ = y.squeeze() if isinstance(y, pd.DataFrame) else y
        self._label_encoder = LabelEncoder().fit(y_)
        if any([isinstance(yi, str) for yi in y_]):
            # If target values are `str` we encode them or scikit-learn will complain.
            y = self._label_encoder.transform(y_)
        self._evaluation_library.determine_sample_indices(stratify=y)
        super().fit(x, y, *args, **kwargs)

    def partial_fit(self, x, y, *args, **kwargs):
        """Should use base class documentation."""
        y_ = y.squeeze() if isinstance(y, pd.DataFrame) else y
        self._label_encoder = LabelEncoder().fit(y_)
        if any([isinstance(yi, str) for yi in y_]):
            # If target values are `str` we encode them or scikit-learn will complain.
            y = self._label_encoder.transform(y_)
        self._evaluation_library.determine_sample_indices(stratify=y)
        super().partial_fit(x, y, *args, **kwargs)

    def _encode_labels(self, y):
        self._label_encoder = LabelEncoder().fit(y)
        return self._label_encoder.transform(y)
