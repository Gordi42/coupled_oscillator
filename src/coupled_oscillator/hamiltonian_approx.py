import coupled_oscillator as co


class HamiltonianApprox(co.Hamiltonian):
    r"""
    The hamiltonian of the system for small perturbations.

    The full hamiltonian is given by:

    .. math::
        H(\boldsymbol \pi, t) = H_0(\boldsymbol \pi, t) + ϵ H_1(\boldsymbol \pi, t)

    with the coupled (:math:`H_0`) and uncoupled (:math:`H_1`) hamiltonians given by:
    
    .. math::
        H_0(\boldsymbol \pi, t) = 
        \frac{p_φ^2}{2 m \ell^2} + \frac{p_ξ^2}{2 m}
        + \frac{k ξ^2}{2} - m g ℓ \cos(φ)

        H_1(\boldsymbol \pi, t) = 
        - \frac{ξ (ξ + 2 \ell) p_φ^2}{2 m \ell^2 (ℓ + ξ)^2}
        - m g ξ \cos(φ) 

    we now do a small perturbation approximation, where we assume that 
    :math:`ξ \ll ℓ` and :math:`φ \ll 1`. Simple taylor expansion gives:

    .. math::
        \cos(φ) ≈ 1 - \frac{φ^2}{2}

        \frac{ξ(ξ + 2 ℓ)}{(ℓ + ξ)^2} ≈ \frac{2 ξ}{ℓ}

    Substituting these into the hamiltonian, we get:

    .. math::
        H_0(\boldsymbol \pi, t) = 
        \frac{p_φ^2}{2 m \ell^2} + \frac{p_ξ^2}{2 m}
        + \frac{k ξ^2}{2} + \frac{m g ℓ φ^2}{2}

        H_1(\boldsymbol \pi, t) = 
        - \frac{ξ p_φ^2}{m ℓ^3} - m g ξ + \frac{m g ξ φ^2}{2}

    Hamilton's equations
    --------------------

    .. math::
        \dot{φ} = \frac{p_φ}{m ℓ^2} - ε \frac{ξ}{m ℓ^3} p_φ

        \dot{ξ} = \frac{p_ξ}{m}

        \dot{p_φ} = - m g ℓ φ - ε m g ξ φ

        \dot{p_ξ} = - k ξ - ε \left( \frac{p_φ^2}{m ℓ^3} + m g - \frac{m g φ^2}{2} \right)
    """
    def kinetic_energy(self, phase: co.Phase, time: float) -> float:
        # get the oscillator properties
        m = self.osc_prop.mass(time)
        ϵ = self.osc_prop.epsilon(time)
        ℓ = self.osc_prop.length
        
        # get the phase properties
        ξ = phase.displacement
        p_φ = phase.angle_momentum
        p_ξ = phase.displacement_momentum

        uncoupled_ekin = 0.5 * p_φ**2 / (m * ℓ**2) + 0.5 * p_ξ**2 / m
        coupled_ekin = - ξ * p_φ**2 / (m * ℓ**3)
        ekin = uncoupled_ekin + ϵ * coupled_ekin
        return ekin

    def potential_energy(self, phase: co.Phase, time: float) -> float:
        # get the oscillator properties
        m = self.osc_prop.mass(time)
        g = self.osc_prop.gravity(time)
        ϵ = self.osc_prop.epsilon(time)
        k = self.osc_prop.spring_constant(time)
        ℓ = self.osc_prop.length
        
        # get the phase properties
        φ = phase.angle
        ξ = phase.displacement

        # compute the coupling potential
        v_coupled = 0.5 * k * ξ**2 + 0.5 * m * g * ℓ * φ**2
        # compute the uncoupling potential
        v_uncoupled = - m * g * ξ + 0.5 * m * g * ξ * φ**2
        # compute the full potential
        epot = v_coupled + ϵ * v_uncoupled
        return epot

    def tendency_angle(
            self, phase: co.Phase, time: float) -> float:
        # get the oscillator properties
        m = self.osc_prop.mass(time)
        ϵ = self.osc_prop.epsilon(time)
        ℓ = self.osc_prop.length

        # get the phase properties
        ξ = phase.displacement
        p_φ = phase.angle_momentum

        # compute uncoupled tendency
        uncoupled_tendency = p_φ / (m * ℓ**2)
        # compute coupled tendency
        coupled_tendency = - ξ / (m * ℓ**3) * p_φ
        # compute full tendency
        dφdt = uncoupled_tendency + ϵ * coupled_tendency
        return dφdt

    def tendency_displacement(
            self, phase: co.Phase, time: float) -> float:
        # get the oscillator properties
        m = self.osc_prop.mass(time)

        # get the phase properties
        p_ξ = phase.displacement_momentum

        # compute tendency
        dξdt = p_ξ / m
        return dξdt

    def tendency_angle_momentum(
            self, phase: co.Phase, time: float) -> float:
        # get the oscillator properties
        m = self.osc_prop.mass(time)
        g = self.osc_prop.gravity(time)
        ϵ = self.osc_prop.epsilon(time)
        ℓ = self.osc_prop.length

        # get the phase properties
        φ = phase.angle
        ξ = phase.displacement

        # compute uncoupled tendency
        uncoupled_tendency = - m * g * ℓ * φ
        # compute coupled tendency
        coupled_tendency = - m * g * ξ * φ
        # compute full tendency
        dpφdt = uncoupled_tendency + ϵ * coupled_tendency
        return dpφdt

    def tendency_displacement_momentum(
            self, phase: co.Phase, time: float) -> float:
        # get the oscillator properties
        m = self.osc_prop.mass(time)
        g = self.osc_prop.gravity(time)
        ϵ = self.osc_prop.epsilon(time)
        k = self.osc_prop.spring_constant(time)
        ℓ = self.osc_prop.length

        # get the phase properties
        φ = phase.angle
        ξ = phase.displacement
        p_φ = phase.angle_momentum

        # compute uncoupled tendency
        uncoupled_tendency = - k * ξ
        # compute coupled tendency
        coupled_tendency = m * g * (1 - φ**2 / 2) + p_φ**2 / (m * ℓ**3)
        # compute full tendency
        dpξdt = uncoupled_tendency + ϵ * coupled_tendency
        return dpξdt
