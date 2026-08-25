"""UnderwaterPreprocessor: White Balance -> CLAHE -> Gamma correction."""

from __future__ import annotations

import cv2
import numpy as np

from ..config import CFG


class UnderwaterPreprocessor:
    def __init__(
        self,
        use_white_balance: bool | None = None,
        use_clahe: bool | None = None,
        use_gamma: bool | None = None,
        gamma_value: float | None = None,
        clahe_clip: float | None = None,
        clahe_tile: tuple[int, int] = (8, 8),
    ):
        self.use_white_balance = CFG.use_white_balance if use_white_balance is None else use_white_balance
        self.use_clahe = CFG.use_clahe if use_clahe is None else use_clahe
        self.use_gamma = CFG.use_gamma if use_gamma is None else use_gamma
        self.gamma_value = CFG.gamma_value if gamma_value is None else gamma_value
        self.clahe_clip = CFG.clahe_clip if clahe_clip is None else clahe_clip
        self.clahe_tile = clahe_tile

        self._clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=self.clahe_tile)

    def white_balance(self, img_bgr: np.ndarray) -> np.ndarray:
        """Gray-world: I'_c = a_c * I_c, a_c = mean_gray / mean_c."""
        img = img_bgr.astype(np.float32)
        means = img.reshape(-1, 3).mean(axis=0)
        gray_mean = means.mean()
        scale = gray_mean / np.clip(means, 1.0, None)
        img = img * scale.reshape(1, 1, 3)
        return np.clip(img, 0, 255).astype(np.uint8)

    def clahe(self, img_bgr: np.ndarray) -> np.ndarray:
        """CLAHE on the L channel in LAB space (preserves color balance)."""
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def gamma_correct(self, img_bgr: np.ndarray) -> np.ndarray:
        """I' = I^gamma, in [0,1] space."""
        img = img_bgr.astype(np.float32) / 255.0
        img = np.power(np.clip(img, 0, 1), self.gamma_value)
        return np.clip(img * 255.0, 0, 255).astype(np.uint8)

    def __call__(self, img_bgr: np.ndarray) -> np.ndarray:
        out = img_bgr
        if self.use_white_balance:
            out = self.white_balance(out)
        if self.use_clahe:
            out = self.clahe(out)
        if self.use_gamma:
            out = self.gamma_correct(out)
        return out
