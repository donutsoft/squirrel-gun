from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import json
from sklearn.linear_model import Ridge, LinearRegression, HuberRegressor
from sklearn.preprocessing import PolynomialFeatures


@dataclass
class LinearAimer:
    # Absolute mapping model:
    # pan  = a0 + a1*u + a2*v
    # tilt = b0 + b1*u + b2*v
    # where u = x/img_w in [0,1], v = y/img_h in [0,1]
    pan: List[float]
    tilt: List[float]
    path: Path | None = None

    @staticmethod
    def default() -> "LinearAimer":
        # Defaults chosen for plausible behavior before training.
        # Pan increases to the right; tilt decreases toward the bottom.
        # Center (u=0.5,v=0.5) ~ (90,90).
        # Use quadratic-compatible coefficient lengths (intercept + 5 terms):
        # features = [u, v, u^2, u*v, v^2]
        # Keep quadratic terms at zero so it behaves like linear initially.
        return LinearAimer(
            pan=[60.0, 60.0, 0.0, 0.0, 0.0, 0.0],
            tilt=[112.5, 0.0, -45.0, 0.0, 0.0, 0.0]
        )

    @staticmethod
    def load(path: Path) -> "LinearAimer":
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            if 'pan' in data and 'tilt' in data:
                pan_list = list(map(float, data['pan']))
                tilt_list = list(map(float, data['tilt']))
                # Backward compatibility: accept 3-coeff (linear) or 6-coeff (quadratic) models
                if (len(pan_list) in (3, 6)) and (len(tilt_list) in (3, 6)):
                    return LinearAimer(pan=pan_list, tilt=tilt_list, path=path)
        except Exception:
            pass
        m = LinearAimer.default()
        m.path = path
        return m

    def save(self) -> None:
        if not self.path:
            return
        try:
            with open(self.path, 'w') as f:
                json.dump({'pan': self.pan, 'tilt': self.tilt}, f)
        except Exception:
            pass

    def to_dict(self) -> Dict[str, List[float]]:
        return {'pan': self.pan, 'tilt': self.tilt}

    def predict(self, u: float, v: float) -> Tuple[float, float]:
        # Predict absolute pan/tilt from normalized coordinates (u,v)
        # Support both linear (3 coeffs) and quadratic (6 coeffs) models.
        if len(self.pan) == 3 and len(self.tilt) == 3:
            a0, a1, a2 = self.pan
            b0, b1, b2 = self.tilt
            pan = a0 + a1 * u + a2 * v
            tilt = b0 + b1 * u + b2 * v
            return pan, tilt

        # Quadratic model: coefficients = [intercept, a_u, a_v, a_uu, a_uv, a_vv]
        # features = [u, v, u^2, u*v, v^2]
        if len(self.pan) == 6 and len(self.tilt) == 6:
            a0, a_u, a_v, a_uu, a_uv, a_vv = self.pan
            b0, b_u, b_v, b_uu, b_uv, b_vv = self.tilt
            uu = u * u
            vv = v * v
            uv = u * v
            pan = a0 + (a_u * u + a_v * v + a_uu * uu + a_uv * uv + a_vv * vv)
            tilt = b0 + (b_u * u + b_v * v + b_uu * uu + b_uv * uv + b_vv * vv)
            return pan, tilt

        # Fallback (unexpected lengths): treat as linear best-effort
        try:
            a0, a1, a2 = self.pan[:3]
            b0, b1, b2 = self.tilt[:3]
            return a0 + a1 * u + a2 * v, b0 + b1 * u + b2 * v
        except Exception:
            return 90.0, 90.0

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

        # Polynomial features (degree=2): [u, v, u^2, u*v, v^2]
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X = poly.fit_transform(X_base)

        # Prefer robust regression to handle occasional bad clicks
        try:
            reg_pan = HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=1000)
            reg_tilt = HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=1000)
            reg_pan.fit(X, y_pan, sample_weight=weights)
            reg_tilt.fit(X, y_tilt, sample_weight=weights)
        except Exception:
            # Fall back to Ridge for stability
            try:
                reg_pan = Ridge(alpha=1.0, fit_intercept=True)
                reg_tilt = Ridge(alpha=1.0, fit_intercept=True)
                reg_pan.fit(X, y_pan, sample_weight=weights)
                reg_tilt.fit(X, y_tilt, sample_weight=weights)
            except Exception:
                # Fallback to simple linear regression if something goes wrong
                lin_pan = LinearRegression(fit_intercept=True)
                lin_tilt = LinearRegression(fit_intercept=True)
                try:
                    lin_pan.fit(X_base, y_pan, sample_weight=weights)
                    lin_tilt.fit(X_base, y_tilt, sample_weight=weights)
                except Exception:
                    lin_pan.fit(X_base, y_pan)
                    lin_tilt.fit(X_base, y_tilt)
                self.pan = [float(lin_pan.intercept_), float(lin_pan.coef_[0]), float(lin_pan.coef_[1])]
                self.tilt = [float(lin_tilt.intercept_), float(lin_tilt.coef_[0]), float(lin_tilt.coef_[1])]
                self.save()
                return

        # Store as [intercept, coef_u, coef_v, coef_u2, coef_uv, coef_v2]
        # PolynomialFeatures order for 2 inputs with include_bias=False is: [u, v, u^2, u*v, v^2]
        pan_coeffs = [float(reg_pan.intercept_)] + [float(c) for c in reg_pan.coef_.tolist()]
        tilt_coeffs = [float(reg_tilt.intercept_)] + [float(c) for c in reg_tilt.coef_.tolist()]

        # Ensure expected length (6). If underdetermined, pad with zeros.
        while len(pan_coeffs) < 6:
            pan_coeffs.append(0.0)
        while len(tilt_coeffs) < 6:
            tilt_coeffs.append(0.0)

        self.pan = pan_coeffs[:6]
        self.tilt = tilt_coeffs[:6]
        self.save()
