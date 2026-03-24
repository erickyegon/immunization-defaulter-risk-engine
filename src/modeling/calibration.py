from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV


def calibrate_model(model, X_valid, y_valid):
    calibrated = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    calibrated.fit(X_valid, y_valid)
    return calibrated
