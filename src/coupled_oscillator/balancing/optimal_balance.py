from copy import deepcopy
import numpy as np
import coupled_oscillator as co


class OptimalBalance(co.balancing.BalanceBase):
    def __init__(self, 
                 hamiltonian: co.Hamiltonian, 
                 linear_balance: co.balancing.BalanceBase,
                 ramp_method: co.ramp_functions.RampFunction,
                 ramp_period: float,
                 max_it: int = 4,
                 stop_criterion: float = 1e-12):
        super().__init__(hamiltonian)
        self.linear_balance = linear_balance
        self.ramp_method = ramp_method
        self.ramp_period = ramp_period
        self.max_it = max_it
        self.stop_criterion = stop_criterion

        # create a ramp function that is one at t=0 and 0 at t=ramp_period
        ramp_func = self.ramp_method(start_time=0,
                                     period=self.ramp_period,
                                     start_value=1,
                                     end_value=0)

        # setup the ramping hamiltonian
        osc_prop = hamiltonian.osc_prop
        osc_prop_ramp = co.OscillatorProperties(
            gravity=osc_prop.gravity,
            mass=osc_prop.mass,
            length=osc_prop.length,
            spring_constant=osc_prop.spring_constant,
            epsilon=ramp_func)
        ramp_hamilton = hamiltonian.__class__(osc_prop_ramp)
        self.ramp_hamilton = ramp_hamilton


    def forward_to_uncoupled(self, phase: co.Phase) -> co.Phase:
        new_phase = co.solve_for_phase(
            hamiltonian=self.ramp_hamilton,
            phase=phase,
            time_span=(0, self.ramp_period))
        return new_phase
                                     
    def backward_to_coupled(self, phase: co.Phase) -> co.Phase:
        new_phase = co.solve_for_phase(
            hamiltonian=self.ramp_hamilton,
            phase=phase,
            time_span=(self.ramp_period, 0))
        return new_phase

    def balance(self, phase: co.Phase, time: float) -> co.Phase:
        lin_bal = self.linear_balance.balance
        iterations = np.arange(self.max_it)
        errors     = np.ones(self.max_it)

        # save the base coordinate
        phase_base = lin_bal(phase, time=time)
        phase_res = deepcopy(phase)

        # start the iterations
        for it in iterations:
            # forward ramping
            phase_forward = self.forward_to_uncoupled(phase_res)
            # project to the base point
            phase_linear = lin_bal(phase_forward, time=time)
            # backward ramping
            phase_nonlin = self.backward_to_coupled(phase_linear)
            # exchange base point coordinate
            phase_new = phase_nonlin - lin_bal(phase_nonlin, time=time) + phase_base

            # calculate the error
            errors[it] = error = phase_new.norm_of_difference(phase_res)

            # update the state
            phase_res = phase_new

            # check the stopping criterion
            if error < self.stop_criterion:
                break

            # check if the error is increasing
            if it > 0 and error > errors[it-1]:
                break

            # recalculate the base coordinate if needed
            if False:
                # check if it is not the last iteration
                if it < self.max_it - 1:
                    phase_base = lin_bal(phase_res, time=time)

        return phase_res

