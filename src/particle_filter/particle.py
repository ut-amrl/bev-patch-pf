from dataclasses import dataclass

from manifpy import SE2


@dataclass(slots=True)
class Particle:
    pose: SE2
    weight: float

    @property
    def x(self) -> float:
        return float(self.pose.coeffs()[0])

    @property
    def y(self) -> float:
        return float(self.pose.coeffs()[1])

    @property
    def theta(self) -> float:
        return float(self.pose.angle())

    @property
    def cos_theta(self) -> float:
        return float(self.pose.coeffs()[2])

    @property
    def sin_theta(self) -> float:
        return float(self.pose.coeffs()[3])
