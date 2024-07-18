from mne.decoding import CSP
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

from .fbcsp import FBCSP


# Define a function to reshape the data for sklearn compatibility
def _reshape_to_2d(X):
    return X.reshape(X.shape[0], -1)


# Create the pipeline
def make_pipeline():
    return Pipeline(
        [
            (
                "csp",
                CSP(
                    component_order="alternate",
                    cov_est="epoch",
                    log=True,
                    n_components=2,
                    # transform_into='csp_space',
                ),
            ),
            # ("fbcsp", FBCSP()),
            # ("scaler", StandardScaler()),
            ("classifier", SVC(C=1, kernel="linear")),
        ]
    )


best_params = {"classifier__C": 1, "classifier__gamma": "scale", "classifier__kernel": "rbf", "csp__n_components": 4}


grid_params = {
    "csp__n_components": [2, 4, 6, 8],
    "classifier__C": [0.1, 1, 10],
    "classifier__gamma": ["scale", "auto", 0.1, 0.01],
    "classifier__kernel": ["rbf", "linear"],
}
