from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import json
from sklearn.linear_model import LinearRegression


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
        return LinearAimer(pan=[60.0, 60.0, 0.0], tilt=[112.5, 0.0, -45.0])

    @staticmethod
    def load(path: Path) -> "LinearAimer":
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            if 'pan' in data and 'tilt' in data and len(data['pan']) == 3 and len(data['tilt']) == 3:
                return LinearAimer(pan=list(map(float, data['pan'])), tilt=list(map(float, data['tilt'])), path=path)
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
        a0, a1, a2 = self.pan
        b0, b1, b2 = self.tilt
        pan = a0 + a1 * u + a2 * v
        tilt = b0 + b1 * u + b2 * v
        return pan, tilt

    def fit_from_clicks(self, rows: List[Dict[str, float]]) -> None:
        # Build feature matrix X = [[u, v], ...] and targets pan/tilt
        X: List[Tuple[float, float]] = []
        y_pan: List[float] = []
        y_tilt: List[float] = []
        for r in rows:
            w = float(r['img_w'])
            h = float(r['img_h'])
            if w <= 0 or h <= 0:
                continue
            u = float(r['x_px']) / w
            v = float(r['y_px']) / h
            X.append((u, v))
            y_pan.append(float(r['pan']))
            y_tilt.append(float(r['tilt']))

        if not X:
            return

        # Fit two independent linear regressions with intercept
        reg_pan = LinearRegression(fit_intercept=True)
        reg_tilt = LinearRegression(fit_intercept=True)
        reg_pan.fit(X, y_pan)
        reg_tilt.fit(X, y_tilt)

        # Coefficients format: [intercept, coef_u, coef_v]
        self.pan = [float(reg_pan.intercept_), float(reg_pan.coef_[0]), float(reg_pan.coef_[1])]
        self.tilt = [float(reg_tilt.intercept_), float(reg_tilt.coef_[0]), float(reg_tilt.coef_[1])]
        self.save()
