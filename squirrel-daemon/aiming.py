from __future__ import annotations

from typing import Tuple

class Calculate_Hose_Angles:
    """Bilinear interpolation from normalized image coords to yaw/pitch (deg).

    Coordinates:
    ----(0,0)  -------------- (1,0)
         |                       |
         |                       |
         |                       |
       (0,1) ---------------- (1,1)
       
    - u in [0,1]: left (0) → right (1)
    - v in [0,1]: top (0) → bottom (1)

    Corner clamps (degrees):
    - Top-Left     (u=0, v=0): Pan 108, Tilt 57
    - Top-Right    (u=1, v=0): Pan 173, Tilt 58
    - Bottom-Left  (u=0, v=1): Pan 110, Tilt 89
    - Bottom-Right (u=1, v=1): Pan 171, Tilt 89

    Returns (yaw, pitch) in degrees, where yaw≈pan and pitch≈tilt.
    """

    # Yaw (pan) degrees at the four image corners
    YAW_TL = 108.0
    YAW_TR = 173.0
    YAW_BL = 110.0
    YAW_BR = 171.0

    # Pitch (tilt) degrees at the four image corners
    PITCH_TL = 57.0
    PITCH_TR = 58.0
    PITCH_BL = 89.0
    PITCH_BR = 89.0

    @staticmethod
    def get_yaw_pitch(u: float, v: float) -> Tuple[float, float]:
        """Compute (yaw, pitch) for normalized (u,v) via bilinear interpolation.

        Inputs outside [0,1] are clamped to the nearest bound.
        """
        # Clamp inputs
        u = 0.0 if u < 0.0 else 1.0 if u > 1.0 else float(u)
        v = 0.0 if v < 0.0 else 1.0 if v > 1.0 else float(v)

        # Bilinear interpolation weights
        w_tl = (1.0 - u) * (1.0 - v)
        w_tr = u * (1.0 - v)
        w_bl = (1.0 - u) * v
        w_br = u * v

        # Interpolate pan (yaw)
        yaw = (
            Calculate_Hose_Angles.YAW_TL * w_tl
            + Calculate_Hose_Angles.YAW_TR * w_tr
            + Calculate_Hose_Angles.YAW_BL * w_bl
            + Calculate_Hose_Angles.YAW_BR * w_br
        )

        # Interpolate tilt (pitch)
        pitch = (
            Calculate_Hose_Angles.PITCH_TL * w_tl
            + Calculate_Hose_Angles.PITCH_TR * w_tr
            + Calculate_Hose_Angles.PITCH_BL * w_bl
            + Calculate_Hose_Angles.PITCH_BR * w_br
        )

        # Return as (yaw, pitch) in degrees
        return float(yaw), float(pitch)

#print(Calculate_Hose_Angles.get_yaw_pitch(0.5,0.5))