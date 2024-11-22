from collections.abc import Callable


class OscillatorProperties:
    """
    The physical properties of the coupled oscillator.

    Parameters
    ----------
    `length` : float
        The length of the spring in the stable position.
    `mass` : float | callable
        The mass of the oscillator (function of time).
    `gravity` : float | callable
        The acceleration due to gravity (function of time).
    `spring_constant` : float | callable
        The spring constant of the oscillator (function of time).
    `epsilon` : float | callable
        A coupling coefficient between the two oscillators (function of time).
    """
    def __init__(self, 
                 length: float = 1,
                 mass: float | Callable = 1, 
                 gravity: float | Callable = 1,
                 spring_constant: float | Callable = 100,
                 epsilon: float | Callable = 1) -> None:
        self.length = length
        self.mass = self.make_callable(mass)
        self.gravity = self.make_callable(gravity)
        self.spring_constant = self.make_callable(spring_constant)
        self.epsilon = self.make_callable(epsilon)

    @staticmethod
    def make_callable(value: float | Callable) -> Callable:
        """
        Converts a constant value to a callable function.

        Parameters
        ----------
        value : float | callable
            The value to convert.

        Returns
        -------
        callable
            The callable function.
        """
        if isinstance(value, Callable):
            return value
        def constant_value(_ = None):
            return value
        return constant_value