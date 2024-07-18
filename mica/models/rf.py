from mne.decoding import CSP
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def make_pipeline():
    return Pipeline([("csp", CSP(log=True)), ("classifier", RandomForestClassifier())])


best_params = {
    "classifier__max_depth": 20,
    "classifier__min_samples_split": 10,
    "classifier__n_estimators": 300,
    "csp__n_components": 8,
}

grid_params = {
    "csp__n_components": [2, 4, 6, 8],
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [None, 10, 20],
    "classifier__min_samples_split": [2, 5, 10],
}
