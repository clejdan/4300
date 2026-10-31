"""Data loading and controlled experiments; these functions do not answer the questions."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
ROOT = Path(__file__).resolve().parent

def load_split(part, split):
    """Return X, y, IDs; IDs and target are never model features."""
    df = pd.read_csv(ROOT / 'data' / part / f'{split}.csv')
    return df.drop(columns=['observation_id', 'target']), df.target, df.observation_id

def load_dictionary(part):
    return pd.read_csv(ROOT / 'data' / part / 'feature_dictionary.csv').set_index('feature')

def load_model(name):
    return joblib.load(ROOT / 'models' / f'{name}.joblib')

def refit_regression(model_id, X_train, y_train, drop_feature=None):
    """Use only after you record your Part I identities; keep the original settings."""
    specs = {'A': Ridge(alpha=300), 'B': LinearRegression(),
             'C': Lasso(alpha=5, max_iter=100000, tol=1e-7)}
    X = X_train.drop(columns=[drop_feature]) if drop_feature else X_train
    return make_pipeline(StandardScaler(), specs[model_id]).fit(X, y_train)

def positive_probability(model, X):
    """Probability of label 1, using the model's actual class ordering."""
    return model.predict_proba(X)[:, list(model.classes_).index(1)]

def predict_at_threshold(model, X, threshold=0.50):
    """Assignment rule: a probability equal to the threshold is positive."""
    return (positive_probability(model, X) >= threshold).astype(int)

def perturb_inputs(X, X_train, dictionary, seed=42, repeats=20, sample_size=50):
    """One required common protocol for Parts III and V.
    Select 50 rows uniformly without replacement. For every numerical feature,
    add independent uniform noise in [-0.05 SD, +0.05 SD]. Round to the nearest
    valid measurement value, then clip to the valid range. SD uses ddof=0 on
    training rows only. Rounding can increase the realized change by half a step.
    Outputs: originals, perturbed rows, source row positions (each repeated 20x).
    Use the same seed and matrices for every candidate. On test data, seed=43.
    """
    if len(X) < sample_size:
        raise ValueError('Need at least 50 observations for the prescribed protocol.')
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(X), size=sample_size, replace=False)
    original = X.iloc[chosen].reset_index(drop=True)
    perturbed = original.loc[original.index.repeat(repeats)].reset_index(drop=True).copy()
    sd = X_train.std(ddof=0)
    for f in X.columns:
        if not pd.api.types.is_numeric_dtype(X[f]):
            continue
        r = dictionary.loc[f]
        noise = rng.uniform(-0.05, 0.05, len(perturbed)) * sd[f]
        value = perturbed[f].to_numpy() + noise
        value = r['minimum'] + np.round((value-r['minimum'])/r['step']) * r['step']
        perturbed[f] = np.clip(value, r['minimum'], r['maximum'])
    return original, perturbed, np.repeat(chosen, repeats)

def flip_rate(model, original, perturbed, repeats=20, threshold=None):
    """Tree native predictions in III; explicit chosen thresholds in V."""
    if threshold is None:
        before, after = model.predict(original), model.predict(perturbed)
    else:
        before = predict_at_threshold(model, original, threshold)
        after = predict_at_threshold(model, perturbed, threshold)
    return float(np.mean(after != np.repeat(before, repeats)))
