import numpy as np
from scipy.integrate import solve_ivp
import xarray as xr
import coupled_oscillator as co


def solve(hamiltonian: co.Hamiltonian, 
          phase: co.Phase, 
          time_span: tuple[float, float],
          num_points: int,
          tolerance: float = 1e-6,
          max_step: float = 0.01) -> xr.Dataset:
    """
    Solve the Hamiltonian equations of motion for the given Hamiltonian and initial phase.

    Parameters
    ----------
    hamiltonian : co.Hamiltonian
        The Hamiltonian of the system.
    phase : co.Phase
        The initial phase space coordinates.
    time_span : tuple[float, float]
        The time span over which to solve the equations.
    num_points : int
        The number of points at which to evaluate the solution.
    tolerance : float, optional
        The relative tolerance of the solver, by default 1e-6.
    max_step : float, optional
        The maximum step size of the solver, by default 0.01.
    
    Returns
    -------
    xr.Dataset
        The solution of the Hamiltonian equations.
    """

    def right_hand_side(t, y):
        """
        Computes the right-hand side of the Hamiltonian equations.

        Parameters
        ----------
        t : float
            The current time.
        y : np.ndarray
            The current phase space coordinates.
        
        Returns
        -------
        np.ndarray
            The right-hand side of the Hamiltonian equations.
        """
        current_phase = co.Phase(hamiltonian.osc_prop, arr=y)
        tendency = hamiltonian.tendency_phase(phase=current_phase, time=t)
        return tendency._arr

    # set the time evaluation points
    t_eval = np.linspace(time_span[0], time_span[1], num_points)

    # solve the Hamiltonian equations
    solution = solve_ivp(fun=right_hand_side, 
                         t_span=time_span, 
                         y0=phase._arr, 
                         t_eval=t_eval, 
                         max_step=max_step,
                         rtol=tolerance,
                         method='RK45')

    # create the solution as an xarray dataset
    solution = xr.Dataset({
        'time': xr.DataArray(
            solution.t, dims='time', 
            attrs={'units': 's'}),
        'angle': xr.DataArray(
            solution.y[0], dims='time', 
            attrs={'units': 'rad', 'long_name': 'Pendulum Angle'}),
        'displacement': xr.DataArray(
            solution.y[1], dims='time',
            attrs={'units': 'm', 'long_name': 'Spring Displacement'}),
        'angle_momentum': xr.DataArray(
            solution.y[2], dims='time',
            attrs={'units': 'kg m² s⁻¹', 'long_name': 'Pendulum Angular Momentum'}),
        'displacement_momentum': xr.DataArray(
            solution.y[3], dims='time',
            attrs={'units': 'kg m s⁻¹', 'long_name': 'Spring Displacement Momentum'}),
    })
    return solution


def solve_for_phase(
    hamiltonian: co.Hamiltonian, 
    phase: co.Phase, 
    time_span: tuple[float, float],
    tolerance: float = 1e-6,
    max_step: float = 0.01) -> co.Phase:
    """
    Solve the Hamiltonian equations of motion for the given Hamiltonian and initial phase.

    Parameters
    ----------
    hamiltonian : co.Hamiltonian
        The Hamiltonian of the system.
    phase : co.Phase
        The initial phase space coordinates.
    time_span : tuple[float, float]
        The time span over which to solve the equations.
    tolerance : float, optional
        The relative tolerance of the solver, by default 1e-6.
    max_step : float, optional
        The maximum step size of the solver, by default 0.01.
    
    Returns
    -------
    co.Phase
        The solution of the Hamiltonian equations.
    """
    solution = solve(hamiltonian, phase, time_span, 2, tolerance, max_step)
    last_phase = solution.isel(time=-1)
    return co.Phase.from_xr(hamiltonian.osc_prop, last_phase)
