import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MaintenanceNoteTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing=0.0):
        self.smoothing = smoothing

    def fit(self, X, y=None):
        X = X.copy()
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        else:
            X = pd.Series(X[:, 0])

        y = np.asarray(y)
        self.global_mean_ = float(np.nanmean(y)) if y.size else 0.5
        self.note_means_ = (
            pd.Series(y, index=X)
            .groupby(level=0)
            .mean()
            .to_dict()
        )
        self.note_counts_ = (
            pd.Series(y, index=X)
            .groupby(level=0)
            .size()
            .to_dict()
        )
        return self

    def transform(self, X):
        X = X.copy()
        if isinstance(X, pd.DataFrame):
            values = X.iloc[:, 0]
        else:
            values = X[:, 0]

        encoded = []
        for value in values:
            if pd.isna(value) or value == "":
                encoded.append(self.global_mean_)
                continue
            count = self.note_counts_.get(value, 0)
            mean = self.note_means_.get(value, self.global_mean_)
            if self.smoothing > 0 and count > 0:
                encoded.append((count * mean + self.smoothing * self.global_mean_) / (count + self.smoothing))
            else:
                encoded.append(mean)
        return np.array(encoded, dtype=float).reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = ["maintenance_note"]
        input_features = np.atleast_1d(input_features)
        return np.array([f"{feature}_encoded" for feature in input_features], dtype=object)
