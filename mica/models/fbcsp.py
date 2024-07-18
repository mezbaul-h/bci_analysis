import numpy as np
from mne.decoding import CSP
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class FBCSP(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=4, n_features_per_band=2):
        self.n_components = n_components
        self.n_features_per_band = n_features_per_band
        self.csps = []
        self.selectors = []

    def fit(self, X, y):
        for band in range(X.shape[1]):  # Assuming X shape is (n_samples, n_bands, n_channels, n_times)
            csp = CSP(n_components=self.n_components, log=True)
            selector = SelectKBest(k=self.n_features_per_band)
            csp_features = csp.fit_transform(X[:, band], y)
            selector.fit(csp_features, y)
            self.csps.append(csp)
            self.selectors.append(selector)
        return self

    def transform(self, X):
        features = []
        for band in range(X.shape[1]):
            csp_features = self.csps[band].transform(X[:, band])
            selected_features = self.selectors[band].transform(csp_features)
            features.append(selected_features)
        return np.hstack(features)
