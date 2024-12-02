from abc import abstractmethod
import numpy as np


class RampFunction:
    r"""
    Ramp function that smoothly transitions from a start value to an end value 
    over a period of time.

    Parameters
    ----------
    start_time : float
        The time at which the ramp function starts (:math:`t_0`).
    period : float
        The period of time over which the ramp function transitions (:math:`T`).
    start_value : float
        The value of the ramp function at the start time (:math:`f_0`).
    end_value : float
        The value of the ramp function at the end time (:math:`f_T`).

    Description
    -----------

    The ramp function :math:`f(t)` should satisfy the following boundary conditions:

    .. math::
        f(t_0) = f_0, \quad f(t_0 + T) = f_T

    Let's assume we have an elementary ramp function $\rho(t)$ that maps the interval $[0, 1]$ to $[0, 1]$:

    .. math::
        \rho(0) = 0, \quad \rho(1) = 1

    Then we can perform linear transformations to obtain a ramp function that 
    satisfies the boundary conditions:

    .. math::
        f(t) = \Tilde{f}(\rho (h(t)))
    
    where:
        \Tilde{f}(\rho) = f_0 + \rho (f_T - f_0)

        h(t) = \frac{t - t_0}{T}
    
    Base classes will implement the elementary ramp function $\rho(t)$.
    """
    def __init__(self, start_time: float, period: float, start_value: float, end_value: float):
        self.start_time = start_time
        self.period = period
        self.start_value = start_value
        self.end_value = end_value

    @abstractmethod
    def elementary_ramp_function(self, t: float) -> float:
        """
        The elementary ramp function that maps the interval [0, 1] to [0, 1].

        Parameters
        ----------
        t : float
            The time at which to evaluate the ramp function.

        Returns
        -------
        float
            The value of the ramp function at time t.
        """
        pass

    def __call__(self, t: float) -> float:
        """
        Evaluates the ramp function at a given time.

        Parameters
        ----------
        t : float
            The time at which to evaluate the ramp function.

        Returns
        -------
        float
            The value of the ramp function at time t.
        """
        h = (t - self.start_time) / self.period
        rho = self.elementary_ramp_function(h)
        return self.start_value + rho * (self.end_value - self.start_value)

def polynomial(order=1):
    class Polynomial(RampFunction):
        def elementary_ramp_function(self, t: float) -> float:
            return t**order
    return Polynomial

class Cosine(RampFunction):
    def elementary_ramp_function(self, t: float) -> float:
        r"""
        Cosine ramp function

        .. math::
            \rho(t) = \frac{1}{2} \left(1 - \cos(\pi t)\right)
        """
        return 0.5 * (1 - np.cos(np.pi * t))
    
class Exponential(RampFunction):
    def elementary_ramp_function(self, t: float) -> float:
        r"""
        Exponential function with zero derivative at the start and end points.

        .. math::
            \rho(t) = \frac{e^{-1/t}}{e^{-1/t} + e^{-1/(1 - t)}}
        """
        t1 = 1 / np.maximum(1e-32, t)
        t2 = 1 / np.maximum(1e-32, 1 - t)
        return np.exp(-t1) / (np.exp(-t1) + np.exp(-t2))

def power(order=3):
    class Power(RampFunction):
        def elementary_ramp_function(self, t: float) -> float:
            return t**order / (t**order + (1 - t)**order)
    return Power
