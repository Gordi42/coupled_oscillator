import numpy as np
import xarray as xr
import coupled_oscillator as co


class Phase:
    def __init__(self, 
                 osc_prop: co.OscillatorProperties,
                 angle: float = 0, 
                 displacement: float = 0,
                 angle_momentum: float = 0, 
                 displacement_momentum: float = 0,
                 arr: np.ndarray | None = None):
        self._osc_prop = osc_prop
        # check if the array is given, if so then set the array and return
        if arr is not None:
            self._arr = arr
            return
        # otherwise, set the values of the phase from the given arguments
        self._arr = np.zeros(4, dtype=np.float64)
        self.angle = angle
        self.displacement = displacement
        self.angle_momentum = angle_momentum
        self.displacement_momentum = displacement_momentum

    @classmethod
    def from_xr(cls, osc_prop: co.OscillatorProperties, xr_data: xr.Dataset) -> 'Phase':
        """
        Creates a Phase object from an xarray dataset.

        Parameters:
            xr_data (xr.Dataset): The xarray dataset containing the phase data.

        Returns:
            Phase: The phase object created from the xarray dataset.
        """
        return Phase(
            osc_prop=osc_prop,
            angle=xr_data.angle.values,
            displacement=xr_data.displacement.values,
            angle_momentum=xr_data.angle_momentum.values,
            displacement_momentum=xr_data.displacement_momentum.values
        )

    def to_xr(self, time) -> xr.Dataset:
        """
        Converts the phase to an xarray dataset.
        """
        ds = xr.Dataset({
            'time': xr.DataArray(
                [time], dims='time', 
                attrs={'units': 's'}),
            'angle': xr.DataArray(
                [self.angle], dims='time', 
                attrs={'units': 'rad', 'long_name': 'Pendulum Angle'}),
            'displacement': xr.DataArray(
                [self.displacement], dims='time',
                attrs={'units': 'm', 'long_name': 'Spring Displacement'}),
            'angle_momentum': xr.DataArray(
                [self.angle_momentum], dims='time',
                attrs={'units': 'kg m² s⁻¹', 'long_name': 'Pendulum Angular Momentum'}),
            'displacement_momentum': xr.DataArray(
                [self.displacement_momentum], dims='time',
                attrs={'units': 'kg m s⁻¹', 'long_name': 'Spring Displacement Momentum'}),
        })
        return ds


    def norm_of_difference(self, other: 'Phase') -> float:
        r"""
        Computes the norm of the difference between the phase and another phase.

        .. math::
            2 \frac{\left| \boldsymbol \pi_1 - \boldsymbol \pi_2 \right|}
            {\left| \boldsymbol \pi_1 \right| + \left| \boldsymbol \pi_2 \right|}

        Parameters:
            other (Phase): The other phase to compute the difference with.

        Returns:
            float: The norm of the difference between the two phases.
        """
        norm = lambda pi: np.linalg.norm(pi._arr)
        return 2 * norm(self - other) / (norm(self) + norm(other))

    # ================================================================
    #  Coordinate Transformation
    # ================================================================
    def to_cartesian_coordinates(self) -> np.ndarray:
        """
        Converts the phase to cartesian coordinates.

        Returns:
            np.ndarray: The cartesian coordinates of the phase (x, y).
        """
        angle = self.angle
        pendulum_length = self._osc_prop.length + self.displacement
        x = np.sin(angle) * pendulum_length
        y = -np.cos(angle) * pendulum_length
        return np.array([x, y], dtype=np.float64)

    # ================================================================
    #  Representation
    # ================================================================
    def __repr__(self) -> str:
        prop = f"φ: {self.angle:.2f}, "
        prop += f"ξ: {self.displacement:.2f}, "
        prop += f"p_φ: {self.angle_momentum:.2f}, "
        prop += f"p_ξ: {self.displacement_momentum:.2f}"
        return f"Phase({prop})"

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def angle(self) -> float:
        """The angle of the pendulum."""
        return self._arr[0]
    
    @angle.setter
    def angle(self, value: float):
        self._arr[0] = value

    @property
    def displacement(self) -> float:
        """The displacement of the spring."""
        return self._arr[1]
    
    @displacement.setter
    def displacement(self, value: float):
        self._arr[1] = value

    @property
    def angle_momentum(self) -> float:
        """The angular momentum of the pendulum."""
        return self._arr[2]
    
    @angle_momentum.setter
    def angle_momentum(self, value: float):
        self._arr[2] = value

    @property
    def displacement_momentum(self) -> float:
        """The displacement momentum of the spring."""
        return self._arr[3]

    @displacement_momentum.setter
    def displacement_momentum(self, value: float):
        self._arr[3] = value
    
    @property
    def osc_prop(self) -> co.OscillatorProperties:
        """The oscillator properties of the phase."""
        return self._osc_prop

    @property
    def arr(self) -> np.ndarray:
        """The array containing the values of the phase."""
        return self._arr
    
    # ================================================================
    #  Operator Overloads
    # ================================================================
    @staticmethod
    def get_value(value):
        """
        Returns the value of the given object.

        If the given object is a Phase object, then the array containing the
        values of the phase is returned. Otherwise, the given object is
        returned as is.
        """
        if isinstance(value, Phase):
            return value._arr
        return value

    def __add__(self, other):
        return Phase(self._osc_prop, arr=self._arr + self.get_value(other))
    
    def __radd__(self, other):
        return Phase(self._osc_prop, arr=self._arr + self.get_value(other))

    def __sub__(self, other):
        return Phase(self._osc_prop, arr=self._arr - self.get_value(other))
    
    def __rsub__(self, other):
        return Phase(self._osc_prop, arr=self.get_value(other) - self._arr)

    def __mul__(self, other):
        return Phase(self._osc_prop, arr=self._arr * self.get_value(other))
    
    def __rmul__(self, other):
        return Phase(self._osc_prop, arr=self._arr * self.get_value(other))
    
    def __truediv__(self, other):
        return Phase(self._osc_prop, arr=self._arr / self.get_value(other))
    
    def __rtruediv__(self, other):
        return Phase(self._osc_prop, arr=self.get_value(other) / self._arr)
    
    def __neg__(self):
        return Phase(self._osc_prop, arr=-self._arr)
    
    def __pos__(self):
        return Phase(self._osc_prop, arr=+self._arr)