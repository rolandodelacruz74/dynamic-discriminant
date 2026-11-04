"""Predictive performance measures used throughout the paper."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def logit(p, eps=1e-9):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def calibration_slope(p, y):
    """Coefficient of the logistic regression of the class on the predicted logit."""
    return LogisticRegression(C=1e6, max_iter=1000).fit(logit(p)[:, None], y).coef_[0, 0]


def summary(p, y, threshold=0.5):
    p = np.asarray(p, float); y = np.asarray(y, int)
    yhat = (p > threshold).astype(int)
    return {"auc": roc_auc_score(y, p), "brier": np.mean((p - y) ** 2), "slope": calibration_slope(p, y),
            "error": np.mean(yhat != y), "sens": np.mean(yhat[y == 1] == 1), "spec": np.mean(yhat[y == 0] == 0)}


_GH = np.polynomial.hermite_e.hermegauss(40)


def expit_normal_mean(mu, sd):
    """E{expit(mu + sd Z)}, Z ~ N(0, 1), by 40-node Gauss-Hermite quadrature (arrays broadcast)."""
    from scipy.special import expit
    mu = np.asarray(mu, float)[..., None]; sd = np.asarray(sd, float)[..., None]
    return (expit(mu + sd * _GH[0]) * _GH[1]).sum(-1) / np.sqrt(2 * np.pi)
