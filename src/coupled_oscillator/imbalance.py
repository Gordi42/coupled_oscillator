import xarray as xr
import coupled_oscillator as co


class Imbalance:
    r"""
    Test balancing operators by computing the diagnosed imbalance.

    Parameters
    ----------
    projector : co.balancing.BalanceBase
        The balancing operator to test.

    Definition of imbalance
    -----------------------

    Let the phase state be given by :math:`\boldsymbol \pi \in \Gamma`, where
    :math:`\Gamma` is the phase space of a dynamical system. We consider a 
    balancing operator :math:`\mathcal B: \Gamma \to \Gamma`, that maps the
    phase state to a balanced state. We define the imbalance as the norm of
    the difference between the unbalanced and balanced phase states:

    .. math::
        \text{Im}(\boldsymbol \pi) = \text{NOD}(\boldsymbol \pi, \mathcal B(\boldsymbol \pi))

    where :math:`\text{NOD}` is the norm of difference between two phase states.

    Definition of diagnosed imbalance
    ---------------------------------

    The procedure to diagnose the imbalance is as follows:
    1. Balance the initial phase state :math:`\boldsymbol \pi_0` at time :math:`t_0`.
    2. Integrate the balanced phase state to a time :math:`t_1`.
    3. Compute the imbalance at time :math:`t_1`.

    We can therefore write the diagnosed imbalance as a function of phase and time:

    .. math::
        \text{DI} = \text{DI}(\boldsymbol \pi_0, t_0, t_1)

    A balancing operator could perform well for a certain diagnose period, but
    may fail for a different diagnose period. To address this, we can integrate
    the imbalance over a time span to get the integrated imbalance:

    .. math::
        \text{IDI} = \int_{t_0}^{t_1} \text{DI}(\boldsymbol \pi_0, t_0, t) dt

    Dividing the integrated imbalance by the time span gives the average imbalance:

    .. math::
        \text{ADI} = \frac{\text{IDI}}{t_1 - t_0}
    """
    def __init__(self, projector: co.balancing.BalanceBase):
        self._projector = projector

    def diagnosed_imbalance(self, 
                            phase: co.Phase, 
                            time_span: tuple[float, float]) -> None:
        """
        Returns the diagnosed imbalance at a given time.

        Parameters
        ----------
        phase : co.Phase
            The initial phase state ot be tested for imbalance.
        time_span : tuple[float, float]
            The time span over which to integrate the phase state.
        
        Returns
        -------
        float
            The imbalance at the final time.
        """
        # we first balance the initial phase state
        start_time, end_time = time_span
        balanced_phase = self.projector.balance(phase, start_time)
        # we integrate the balanced phase state to the final time
        solution = co.solve(hamiltonian=self.hamiltonian,
                            phase=balanced_phase,
                            time_span=time_span,
                            num_points=2)
        # compute the imbalance at the final time
        imbalance = self._imbalance_time_step(solution.isel(time=-1))
        return imbalance.imbalance.item()

    def diagnosed_imbalance_time_series(
            self, 
            phase: co.Phase, 
            time_span: tuple[float, float], 
            num_points: int) -> xr.Dataset:
        """
        Returns a time series of the diagnosed imbalance over a time span.

        Parameters
        ----------
        phase : co.Phase
            The initial phase state to be tested for imbalance.
        time_span : tuple[float, float]
            The time span over which to integrate the phase state.
        num_points : int
            The number of points at which to evaluate the solution.

        Returns
        -------
        xr.Dataset
            The time series of the diagnosed imbalance.
        """
        # we first balance the initial phase state
        start_time, end_time = time_span
        balanced_phase = self.projector.balance(phase, start_time)
        # we integrate the balanced phase state to each time step
        solution = co.solve(hamiltonian=self.hamiltonian,
                            phase=balanced_phase,
                            time_span=time_span,
                            num_points=num_points)
        # compute the imbalance at each time step
        imbalance = solution.groupby("time").map(self._imbalance_time_step)
        return imbalance

    def integrated_imbalance(self, 
                             phase: co.Phase, 
                             time_span: tuple[float, float],
                             num_points: int) -> None:
        """
        Computes the integrated imbalance

        Parameters
        ----------
        phase : co.Phase
            The initial phase state to be tested for imbalance.
        time_span : tuple[float, float]
            The time span over which to integrate the phase state.
        num_points : int
            The number of points at which to evaluate the solution.

        Returns
        -------
        float
            The integrated imbalance over the time span.
        """
        dt = (time_span[1] - time_span[0]) / num_points
        # get the imbalance time series
        imbalance = self.diagnosed_imbalance_time_series(phase, time_span, num_points)
        # multiply the imbalance with the time step and sum over all time steps
        integrated_imbalance = (imbalance["imbalance"] * dt).sum().values
        return integrated_imbalance

    def average_imbalance(self,
                          phase: co.Phase,
                          time_span: tuple[float, float],
                          num_points: int) -> None:
        """
        Computes the average imbalance

        Parameters
        ----------
        phase : co.Phase
            The initial phase state to be tested for imbalance.
        time_span : tuple[float, float]
            The time span over which to integrate the phase state.
        num_points : int
            The number of points at which to evaluate the solution.

        Returns
        -------
        float
            The average imbalance over the time span.
        """
        integrated_imbalance = self.integrated_imbalance(phase, time_span, num_points)
        # divide the integrated imbalance by the time span
        average_imbalance = integrated_imbalance / (time_span[1] - time_span[0])
        return average_imbalance

    def _imbalance_time_step(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Computes the imbalance at a single time step of a xarray dataset.

        Parameters
        ----------
        ds : xr.Dataset
            The time step of phase space coordinates. 
            (Must contain only one time step)

        Returns
        -------
        xr.Dataset
            The imbalance at the time step (stored in the "imbalance" variable).
        """
        unbalanced_phase = co.Phase.from_xr(osc_prop=self.osc_prop, xr_data=ds)
        balanced_phase = self.projector.balance(unbalanced_phase, ds["time"].item())
        # compute the norm of difference between the unbalanced and balanced phase
        imbalance = unbalanced_phase.norm_of_difference(balanced_phase)
        # we create a dataset with the imbalance
        imbalance_ds = xr.Dataset({
            'time': xr.DataArray(
                [ds["time"].item()], dims='time',
                attrs={'units': 's', 'long_name': 'Time'}),
            'imbalance': xr.DataArray(
                [imbalance], dims='time',
                attrs={'long_name': 'Imbalance'})
        })
        return imbalance_ds

    # ================================================================
    #  Properties
    # ================================================================

    @property
    def projector(self) -> co.balancing.BalanceBase:
        """The projector that balances the phase space coordinates."""
        return self._projector

    @property
    def hamiltonian(self) -> co.Hamiltonian:
        """The Hamiltonian of the coupled oscillator system."""
        return self.projector.hamiltonian

    @property
    def osc_prop(self) -> co.OscillatorProperties:
        """The properties of the oscillators."""
        return self.hamiltonian.osc_prop