from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def make_pipeline():
    return Pipeline([("csp", CSP(log=True)), ("classifier", LinearDiscriminantAnalysis())])


best_params = {"classifier__shrinkage": 0.9, "classifier__solver": "lsqr", "csp__n_components": 4}

grid_params = {
    "csp__n_components": [2, 4, 6, 8],
    "classifier__solver": ["lsqr", "eigen"],
    "classifier__shrinkage": [None, "auto", 0.1, 0.5, 0.9],
}
