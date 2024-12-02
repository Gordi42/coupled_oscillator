import numpy as np
import coupled_oscillator as co


class GravityEquilibrium(co.balancing.BalanceBase):
    r"""
    Balancing by setting the displacement to the equilibrium position due to gravity.

    Let's consider the :py:class:`hamiltonian <coupled_oscillator.hamiltonian.Hamiltonian>` 
    tendency equations for the displacement:

    .. math::
        \dot{ξ} = \frac{p_ξ}{m}

        \dot{p}_ξ = - k ξ + ϵ m g \cos(\phi) + ϵ \frac{p_φ^2}{m (ℓ + ξ)^3}

    We neglect the third term in the second equation, as it is a small coupling term.
    We than set :math:`p_ξ = 0` and solve the second equation for :math:`ξ`,
    such that :math:`\dot{p}_ξ = 0`:

    .. math::
        ξ = \frac{ϵ m g \cos(\phi)}{k}
    """
    def balance(self, phase: co.Phase, time: float) -> co.Phase:
        # get the oscillator properties
        m = self.osc_prop.mass(time)
        g = self.osc_prop.gravity(time)
        k = self.osc_prop.spring_constant(time)

        # Compute the new displacement due to gravity
        new_displacement = g * m * np.cos(phase.angle) / k
        # Set the new displacement momentum to zero
        new_displacement_momentum = 0
        balanced = co.Phase(osc_prop=self.osc_prop,
                            angle=phase.angle,
                            angle_momentum=phase.angle_momentum,
                            displacement=new_displacement,
                            displacement_momentum=new_displacement_momentum)
        return balanced