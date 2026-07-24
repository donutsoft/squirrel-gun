from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import json
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import PolynomialFeatures


@dataclass
class LinearAimer:
    # Absolute polynomial mapping from normalized image coordinates (u, v)
    # to pan/tilt. Older three- and six-coefficient models remain supported;
    # dense automatic calibration uses ten-coefficient cubic models.
    pan: List[float]
    tilt: List[float]
    path: Path | None = None

    @staticmethod
    def default() -> "LinearAimer":
        # Bootstrap from the installation's known approximate field of view.
        # Automatic calibration immediately replaces this seed with observations.
        # Coefficients are [1, u, v, u^2, u*v, v^2].
        return LinearAimer(
            pan=[108.0, 65.0, 2.0, 0.0, -4.0, 0.0],
            tilt=[57.0, 1.0, 32.0, 0.0, -1.0, 0.0],
        )

    @staticmethod
    def load(path: Path) -> "LinearAimer":
        with open(path, 'r') as f:
            data = json.load(f)
        if 'pan' not in data or 'tilt' not in data:
            raise ValueError(f"Aim model must include pan and tilt coefficients: {path}")
        pan_list = list(map(float, data['pan']))
        tilt_list = list(map(float, data['tilt']))
        # Backward compatibility: linear, quadratic, and cubic models.
        if (len(pan_list) in (3, 6, 10)) and (len(tilt_list) in (3, 6, 10)):
            return LinearAimer(pan=pan_list, tilt=tilt_list, path=path)
        raise ValueError(
            f"Aim model coefficient lengths must be 3, 6, or 10: "
            f"pan={len(pan_list)}, tilt={len(tilt_list)}"
        )

    def save(self) -> None:
        if not self.path:
            return
        with open(self.path, 'w') as f:
            json.dump({'pan': self.pan, 'tilt': self.tilt}, f)

    def to_dict(self) -> Dict[str, List[float]]:
        return {'pan': self.pan, 'tilt': self.tilt}

    def predict(self, u: float, v: float) -> Tuple[float, float]:
        # Predict absolute pan/tilt from normalized coordinates (u,v).
        if len(self.pan) == 3 and len(self.tilt) == 3:
            a0, a1, a2 = self.pan
            b0, b1, b2 = self.tilt
            pan = a0 + a1 * u + a2 * v
            tilt = b0 + b1 * u + b2 * v
            return pan, tilt

        if len(self.pan) == len(self.tilt) and len(self.pan) in (6, 10):
            features = [1.0, u, v, u * u, u * v, v * v]
            if len(self.pan) == 10:
                features.extend([u * u * u, u * u * v, u * v * v, v * v * v])
            pan = sum(coef * feature for coef, feature in zip(self.pan, features))
            tilt = sum(coef * feature for coef, feature in zip(self.tilt, features))
            return float(pan), float(tilt)

        raise ValueError(
            f"Unexpected coefficient lengths: pan={len(self.pan)}, tilt={len(self.tilt)}"
        )

    def fit_from_clicks(self, rows: List[Dict[str, float]], *, focus: Tuple[float, float] | None = None, sigma: float = 0.2) -> None:
        # Build feature matrix X = [[u, v], ...] and targets pan/tilt
        X_base: List[Tuple[float, float]] = []
        y_pan: List[float] = []
        y_tilt: List[float] = []
        weights: List[float] = []
        for r in rows:
            try:
                w = float(r['img_w'])
                h = float(r['img_h'])
                if w <= 0 or h <= 0:
                    continue
                u = float(r['x_px']) / w
                v = float(r['y_px']) / h
                X_base.append((u, v))
                y_pan.append(float(r['pan']))
                y_tilt.append(float(r['tilt']))
                if focus is not None and 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0:
                    u0, v0 = focus
                    du = (u - float(u0))
                    dv = (v - float(v0))
                    d2 = du * du + dv * dv
                    s2 = max(1e-6, float(sigma) * float(sigma))
                    wgt = float(__import__('math').exp(-0.5 * d2 / s2))
                else:
                    wgt = 1.0
                weights.append(wgt)
            except Exception:
                continue

        if not X_base:
            return

        # Once enough observations exist, cubic terms capture lens and linkage
        # non-linearity that a four-corner or quadratic mapping cannot.
        degree = 3 if len(X_base) >= 20 else 2
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X = poly.fit_transform(X_base)

        reg_pan = HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=1000)
        reg_tilt = HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=1000)
        reg_pan.fit(X, y_pan, sample_weight=weights)
        reg_tilt.fit(X, y_tilt, sample_weight=weights)

        # PolynomialFeatures order is:
        # degree 2: [u, v, u^2, u*v, v^2]
        # degree 3 adds [u^3, u^2*v, u*v^2, v^3].
        pan_coeffs = [float(reg_pan.intercept_)] + [float(c) for c in reg_pan.coef_.tolist()]
        tilt_coeffs = [float(reg_tilt.intercept_)] + [float(c) for c in reg_tilt.coef_.tolist()]

        expected_length = 10 if degree == 3 else 6
        while len(pan_coeffs) < expected_length:
            pan_coeffs.append(0.0)
        while len(tilt_coeffs) < expected_length:
            tilt_coeffs.append(0.0)

        self.pan = pan_coeffs[:expected_length]
        self.tilt = tilt_coeffs[:expected_length]
        self.save()
