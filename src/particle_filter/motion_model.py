from collections import deque

import numpy as np
from manifpy import SE2, SE2Tangent

from .particle import Particle


class MotionModel:
    def __init__(
        self, action_gain: tuple[float, float, float], drift_gain: tuple[float, float, float], **kwargs: dict[str, any]
    ) -> None:
        self.action_gain = np.asarray(action_gain, dtype=np.float32)
        self.drift_gain = np.asarray(drift_gain, dtype=np.float32)
        self.action_buffer = deque(maxlen=5)

    def __call__(self, particles: list[Particle], action: np.ndarray, **kwargs):
        """
        Args:
            particles: list of Particle objects to be updated in-place
            action: (3,) array of [delta_x, delta_y, delta_theta]
        """
        sigma = self.action_gain * np.abs(action)

        self.action_buffer.append(action.copy())

        if len(self.action_buffer) >= 2:
            action_mean = np.mean(np.abs(self.action_buffer), axis=0)
            action_std = np.std(self.action_buffer, axis=0)
            cv = action_std / (action_mean + 1e-6)
            sigma *= 1.0 + self.drift_gain * cv

        action_SE2 = SE2(*action)
        for p in particles:
            noise = SE2Tangent(sigma * np.random.normal(size=3))
            p.pose = p.pose * (action_SE2 + noise)
