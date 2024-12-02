import coupled_oscillator as co


class NaiveBalancer(co.balancing.BalanceBase):
    """
    A naive balancing approach that simply sets the displacement to zero.

    This approach does not only ignores the nonlinear coupling between the
    oscillators, but also the effect of gravity on the spring. It is expected
    to have large errors in the balanced state.
    """
    def balance(self, phase: co.Phase, time: float) -> co.Phase:
        balanced = co.Phase(osc_prop=self.osc_prop,
                            # copy the angles from the unbalanced phase
                            angle=phase.angle,
                            angle_momentum=phase.angle_momentum,
                            # set displacement to zero
                            displacement=0,
                            displacement_momentum=0)
        return balanced
