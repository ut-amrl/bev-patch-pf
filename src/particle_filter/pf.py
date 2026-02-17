from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from manifpy import SE2, SE2Tangent

from particle_filter import Particle

if TYPE_CHECKING:
    from particle_filter import MotionModel, ObservationModel


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def belief_changed(require_init: bool = True) -> callable:
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            if require_init and not self.is_initialized:
                raise RuntimeError("Particle filter is not initialized")
            try:
                return fn(self, *args, **kwargs)
            finally:
                self._pose_valid = False

        return wrapper

    return decorator


class ParticleFilter:
    def __init__(
        self,
        motion_model: MotionModel,
        observation_model: ObservationModel,
        n_particles: int,
        n_eff_threshold: float,
        resample_noise_sigma: tuple[float, float, float],
        **kwargs,
    ):
        self.motion_model = motion_model
        self.observation_model = observation_model
        self.particles: list[Particle] = []

        self.n_particles = max(1, int(n_particles))
        self.n_eff_threshold = np.clip(n_eff_threshold, 0.0, 1.0)
        self.resample_noise_sigma = np.asarray(resample_noise_sigma, dtype=np.float32)

        self._pose = np.zeros(3)
        self._pose_valid = False

    @property
    def pose(self) -> np.ndarray:
        if self._pose_valid:
            return self._pose

        if not self.particles:
            logger.error("Particles are not initialized")
            return np.zeros(3)

        x = sum(p.x * p.weight for p in self.particles)
        y = sum(p.y * p.weight for p in self.particles)
        cos_theta = sum(p.cos_theta * p.weight for p in self.particles)
        sin_theta = sum(p.sin_theta * p.weight for p in self.particles)

        if np.hypot(cos_theta, sin_theta) < 1e-6:
            theta = self._pose[2]  # keep previous angle
        else:
            theta = np.arctan2(sin_theta, cos_theta)

        self._pose_valid = True
        self._pose = np.array([x, y, theta])
        return self._pose

    @property
    def is_initialized(self) -> bool:
        return len(self.particles) > 0

    @belief_changed(require_init=False)
    def reset(self) -> None:
        self.particles = []

    @belief_changed(require_init=False)
    def reset_weights(self) -> None:
        weight = 1 / self.n_particles
        for p in self.particles:
            p.weight = weight

    @belief_changed(require_init=False)
    def initialize(self, init_xyr: np.ndarray, init_noise_sigma: np.ndarray = np.zeros(3)) -> None:
        init_pose = SE2(*init_xyr)
        weight = 1 / self.n_particles

        self.particles = []
        self.particles.append(Particle(pose=init_pose, weight=weight))
        for _ in range(self.n_particles - 1):
            noise = SE2Tangent(np.random.normal(size=3) * init_noise_sigma)
            self.particles.append(Particle(pose=init_pose + noise, weight=weight))

    @belief_changed()
    def predict(self, action: np.ndarray, sigma: np.ndarray | None = None) -> None:
        self.motion_model(self.particles, action=action, sigma=sigma)

    @belief_changed()
    def update(
        self, ground_image: torch.Tensor, ground_depth: torch.Tensor, info: dict, results: dict | None = None
    ) -> None:
        # compute likelihood of particles
        log_likelihood = self.observation_model(
            self.particles, ground_image, ground_depth, info, est_xyr=self.pose, results=results
        )

        # update particle weights (posterior = prior * likelihood) with log-sum-exp normalization
        w_prev = np.array([p.weight for p in self.particles], dtype=np.float64)
        logw = np.log(w_prev + 1e-30) + log_likelihood.astype(np.float64)

        m = np.max(logw)
        logw = logw - (m + np.log(np.sum(np.exp(logw - m))))
        for p, lw in zip(self.particles, logw):
            p.weight = np.exp(lw)

    @belief_changed()
    def resample(self) -> bool:
        """Low variance resampling"""
        # compute effective sample size
        n_eff = 1.0 / sum(p.weight**2 for p in self.particles)
        if n_eff > self.n_eff_threshold * self.n_particles:
            return False

        # resampling
        new_particles = []
        weights = [p.weight for p in self.particles]

        M = len(weights)
        r = np.random.uniform(0, 1 / M)
        c = weights[0]
        i = 0
        for m in range(M):
            U = r + m / M
            while c < U and i < M - 1:
                i += 1
                c += weights[i]
            noise = SE2Tangent(np.random.normal(size=3) * self.resample_noise_sigma)
            pose = self.particles[i].pose + noise
            new_particles.append(Particle(pose=pose, weight=1 / self.n_particles))

        self.particles = new_particles
        return True
