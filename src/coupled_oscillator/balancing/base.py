from abc import abstractmethod
import xarray as xr
import coupled_oscillator as co


class BalanceBase:
    """
    An abstract base class for balancing phase space coordinates.

    A balancing operator is a function that maps a state of the coupled
    oscillator system to a balanced state. A balanced state is a state where
    the fast oscillations of the spring are eliminated.

    Child classes must implement the balance method.
    """
    def __init__(self, hamiltonian: co.Hamiltonian) -> None:
        self.hamiltonian = hamiltonian
        self.osc_prop = hamiltonian.osc_prop

    @abstractmethod
    def balance(self, phase: co.Phase, time: float) -> co.Phase:
        """
        Balances the phase space coordinates at a given time.

        Parameters
        ----------
        phase : co.Phase
            The phase space coordinates to balance.
        time : float
            The time at which to balance the phase space coordinates.

        Returns
        -------
        co.Phase
            The balanced phase space coordinates.
        """

    def _balance_time_step(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Balances a single time step of phase space coordinates.

        Parameters
        ----------
        ds : xr.Dataset
            The time step of phase space coordinates.

        Returns
        -------
        xr.Dataset
            The balanced time step of phase space coordinates.
        """
        # Extract the values
        time = ds["time"].item()
        angle = ds["angle"].item()
        displacement = ds["displacement"].item()
        angle_momentum = ds["angle_momentum"].item()
        displacement_momentum = ds["displacement_momentum"].item()

        # Create a phase object
        phase = co.Phase(
            osc_prop=self.osc_prop,
            angle=angle,
            displacement=displacement,
            angle_momentum=angle_momentum,
            displacement_momentum=displacement_momentum,
        )

        # Balance the phase
        balanced_phase = self.balance(phase=phase, time=time)

        # Create a dataset with the balanced phase
        balanced_ds = xr.Dataset({
            "time": xr.DataArray(
                [time], dims="time", 
                attrs=ds["time"].attrs),
            "angle": xr.DataArray(
                [balanced_phase.angle], dims="time", 
                attrs=ds["angle"].attrs),
            "displacement": xr.DataArray(
                [balanced_phase.displacement], dims="time", 
                attrs=ds["displacement"].attrs),
            "angle_momentum": xr.DataArray(
                [balanced_phase.angle_momentum], dims="time", 
                attrs=ds["angle_momentum"].attrs),
            "displacement_momentum": xr.DataArray(
                [balanced_phase.displacement_momentum], dims="time", 
                attrs=ds["displacement_momentum"].attrs),
        })
        return balanced_ds

    def balance_time_series(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Balances a time series of phase space coordinates.

        Parameters
        ----------
        df : xr.Dataset
            The time series of phase space coordinates.

        Returns
        -------
        xr.Dataset
            The balanced time series of phase space coordinates.
        """
        return ds.groupby("time").map(self._balance_time_step)
