import numpy as np
import coupled_oscillator as co


class Hamiltonian:
    r"""
    The hamiltonian of a coupled oscillator. The coupled oscillator
    consist of mass attached to a spring. The movement of the mass
    is two dimensional, so that the angle of the spring can also
    change.

    .. image:: /_static/oscillator.svg
        :width: 40%
        :align: center

    Derivation
    ----------

    Generalized Coordinates
    ~~~~~~~~~~~~~~~~~~~~~~~
    We describe the position of the mass with an angle :math:`\phi` 
    and the displacement of the spring with :math:`\xi`.
    The cartesian coordinates of the mass are then given by:

    .. math::
        \boldsymbol r(t) 
        = \left( \begin{matrix} x(t) \\ y(t) \end{matrix} \right)
        = \left( \begin{matrix} 
            (ℓ + \xi(t)) \sin(\phi(t)) \\ 
        -(ℓ + \xi(t)) \cos(\phi(t)) 
        \end{matrix} \right)

    where :math:`\ell` is the rest length of the spring. 
    We transform our system from the cartesian coordinates 
    :math:`(x, y)` to the generalized coordinates :math:`(\phi, \xi)`.

    Forces and Potentials
    ~~~~~~~~~~~~~~~~~~~~~

    The spring force is given by:

    .. math::
        \boldsymbol F_{\text{spring}} 
        = -k ξ \frac{\boldsymbol r}{\| \boldsymbol r \|}

    where :math:`k(t)` is a time dependent spring constant. Using
    :math:`F = -∇ V` we can find the potential of the spring:

    .. math::
        V_{\text{spring}}(\xi, t) = \frac{1}{2} k \xi^2

    In addition to the spring force, there is a gravitational force 
    acting on the mass. The gravitational potential is given by:

    .. math::
        V_{\text{gravity}}(\phi, \xi, t) = m g y = -m g (ℓ + \xi) \cos(\phi)

    where :math:`m(t)` and :math:`g(t)` are time dependent mass and 
    gravitational acceleration, respectively.

    Kinetic Energy
    ~~~~~~~~~~~~~~
    To find the hamiltonian, we need to find the kinetic energy. 
    The kinetic energy is given by:

    .. math::
        T = \frac{1}{2} m \dot{\boldsymbol r}^2 
        = \frac{1}{2} m \left( \dot{\xi}^2 + (ℓ + \xi)^2 \dot{\phi}^2 \right)

    where the dot denotes the time derivative:

    .. math::
        \dot{f} = \frac{d}{d t} f

    Generalised Momenta
    ~~~~~~~~~~~~~~~~~~~
    The last step before we can write down the hamiltonian 
    is to find the generalised momenta. These are defined as
    the partial derivatives of the lagrangian :math:`L = T - V`
    with respect to the generalised velocities:

    .. math::
        p_φ = \frac{\partial L}{∂ \dot{\phi}} = m (ℓ + \xi)^2 \dot{\phi}

        p_ξ = \frac{\partial L}{∂ \dot{\xi}} = m \dot{\xi}

    Hamiltonian
    ~~~~~~~~~~~
    The hamiltonian of our system is given by the sum of the kinetic
    and potential energy:

    .. math::
        H(\boldsymbol \pi, t) = T + V = 
        \frac{p_\phi^2}{2 m (ℓ + \xi)^2} + \frac{p_\xi^2}{2 m}
        + \frac{k \xi^2}{2} - m g (ℓ + \xi) \cos(\phi)

    where :math:`\boldsymbol π = (\phi, \xi, p_\phi, p_\xi)` is the
    phase vector of the system.
    We rewrite the hamiltonian and sort it by coupled and uncoupled terms:

    .. math::
        H(\boldsymbol \pi, t) = H_0(\boldsymbol \pi, t) + ϵ H_1(\boldsymbol \pi, t)

    where :math:`\epsilon` is a scaling factor that determines the strength of the coupling.
    For :math:`ϵ = 0` we get the uncoupled hamiltonian and for :math:`ϵ = 1`
    we get the fully coupled hamiltonian. The uncoupled part is given by:

    .. math::
        H_0(\boldsymbol \pi, t) = 
        \frac{p_\phi^2}{2 m \ell^2} + \frac{p_\xi^2}{2 m}
        + \frac{k \xi^2}{2} - m g ℓ \cos(\phi)

    The coupled part is given by:

    .. math::
        H_1(\boldsymbol \pi, t) = 
        - \frac{\xi (ξ + 2 \ell) p_\phi^2}{2 m \ell^2 (ℓ + \xi)^2}
        - m g ξ \cos(\phi) 

    Due to the high nonlinearity of the coupled part, the equation becomes
    hard to analyze. It can be simplified by assuming that the spring displacement
    is small compared to the rest length of the spring. And that the angle is small.
    This approximated hamiltonian is detailed in [TODO].
  
    Hamilton's Equations
    --------------------

    The tendency equations are given by the time derivative of the phase vector:

    .. math::
        \dot{\boldsymbol \pi} = \frac{d}{d t} \boldsymbol π

    using the hamiltonian, we can write the tendency equations as:

    .. math::
        \dot{\phi} 
        = \frac{\partial H}{∂ p_\phi} 
        = \frac{p_\phi}{m \ell^2} - ϵ \frac{\xi (ξ + 2 \ell)}{m \ell^2 (ℓ + \xi)^2} p_φ

        \dot{\xi} 
        = \frac{\partial H}{∂ p_\xi} 
        = \frac{p_\xi}{m}

        \dot{p_\phi} = - \frac{\partial H}{∂ \phi} 
            = - m g ℓ \sin(\phi) - ϵ m g ξ \sin(\phi)

        \dot{p_\xi} = - \frac{\partial H}{∂ \xi}
            = - k ξ + ϵ m g \cos(\phi) + ϵ \frac{p_\phi^2}{m (ℓ + \xi)^3} 

    Uncoupled Solution
    ------------------
    We set :math:`ϵ = 0` to obtain the uncoupled system, further 
    we assume small angles :math:`\phi`, such that

    .. math::
        \sin(\phi) ≈ φ 

    this will obtain the hamiltonian equations:

    .. math::
        \dot{\phi} = \frac{p_\phi}{m \ell^2}
          ,  
        \dot{\xi} = \frac{p_\xi}{m}
          ,  
        \dot{p_\phi} = -mgℓ φ
          ,  
        \dot{p_\xi} = -k ξ 

    Combining them yields:

    .. math::
        \ddot{\phi} = -\omega_\phi^2 φ
          ,  
        \ddot{\xi} = -\omega_\xi^2 ξ

    with frequencies:

    .. math::
        \omega_\phi^2 = \frac{g}{\ell}
          ,  
        \omega_\xi^2 = \frac{k}{m}

    these are the well known frequencies of the string pendulum 
    and of the spring pendulum.

    """
    def __init__(self, osc_prop: co.OscillatorProperties) -> None:
        self._osc_prop = osc_prop

    # ================================================================
    #  Energy
    # ================================================================

    def kinetic_energy(self, phase: co.Phase, time: float) -> float:
        r"""
        The kinetic energy of the coupled oscillator.

        .. math::
            T = \frac{1}{2} \frac{p_φ^2}{m ℓ^2} + \frac{1}{2} \frac{p_ξ^2}{m}
              - ε\frac{ξ (ξ + 2 ℓ) p_φ^2}{2 m ℓ^2 (ℓ + ξ)^2}

        Parameters
        ----------
        phase : co.Phase
            The phase :math:`\boldsymbol π` of the coupled oscillator.
        time : float
            The current time :math:`t`.
        
        Returns
        -------
        float
            The kinetic energy :math:`T`.
        """
        # get the oscillator properties
        m = self.osc_prop.mass(time)
        ϵ = self.osc_prop.epsilon(time)
        ℓ = self.osc_prop.length
        
        # get the phase properties
        ξ = phase.displacement
        p_φ = phase.angle_momentum
        p_ξ = phase.displacement_momentum

        # compute the kinetic energy
        uncoupled_ekin = 0.5 * p_φ**2 / (m * ℓ**2) + 0.5 * p_ξ**2 / m
        coupled_ekin = - ξ * (ξ + 2 * ℓ) * p_φ**2 / (2 * m * ℓ**2 * (ℓ + ξ)**2)
        ekin = uncoupled_ekin + ϵ * coupled_ekin
        return ekin

    def potential_energy(self, phase: co.Phase, time: float) -> float:
        r"""
        The potential energy of the coupled oscillator.
        
        .. math::
            V = V_{\text{spring}} + V_{\text{gravity}}

            V_{\text{spring}} = \frac{1}{2} k ξ^2

            V_{\text{gravity}} = - m g (ℓ + ϵ * 
            ξ) \cos(φ)

        Parameters
        ----------
        phase : co.Phase
            The phase :math:`\boldsymbol π` of the coupled oscillator.
        time : float
            The current time :math:`t`.
        
        Returns
        -------
        float
            The potential energy :math:`V`.
        """
        # get the oscillator properties
        m = self.osc_prop.mass(time)
        g = self.osc_prop.gravity(time)
        ϵ = self.osc_prop.epsilon(time)
        k = self.osc_prop.spring_constant(time)
        ℓ = self.osc_prop.length
        
        # get the phase properties
        φ = phase.angle
        ξ = phase.displacement

        # compute the potential of the spring
        v_spring = 0.5 * k * ξ**2
        # compute the potential of the gravity
        v_gravity = - m * g * (ℓ + ϵ * ξ) * np.cos(φ)
        # compute the full potential
        epot = v_spring + v_gravity
        return epot

    def total_energy(self, phase: co.Phase, time: float) -> float:
        r"""
        The total energy of the coupled oscillator (hamiltonian).

        .. math::
            E = H = T + V

        where :math:`T` is the kinetic energy and :math:`V` is the potential energy.

        Parameters
        ----------
        phase : co.Phase
            The phase :math:`\boldsymbol π` of the coupled oscillator.
        time : float
            The current time :math:`t`.
        
        Returns
        -------
        float
            The total energy :math:`E`.
        """
        ekin = self.kinetic_energy(phase, time)
        epot = self.potential_energy(phase, time)
        etot = ekin + epot
        return etot

    # ================================================================
    #  Tendencies
    # ================================================================

    def tendency_angle(
            self, phase: co.Phase, time: float) -> float:
        r"""
        Compute the tendency of the angle of the coupled oscillator.

        .. math::
            \dot{φ} = \frac{p_φ}{m ℓ^2} - ϵ \frac{ξ (ξ + 2 ℓ)}{m ℓ^2 (ℓ + ξ)^2} p_φ

        Parameters
        ----------
        phase : co.Phase
            The phase :math:`\boldsymbol π` of the coupled oscillator.
        time : float
            The current time :math:`t`.
        
        Returns
        -------
        float
            The tendency of the angle :math:`\dot{φ}`.
        """
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
        coupled_tendency = - ξ * (ξ + 2 * ℓ) * p_φ / (m * ℓ**2 * (ℓ + ξ)**2)
        # compute full tendency
        dφdt = uncoupled_tendency + ϵ * coupled_tendency
        return dφdt
    
    def tendency_displacement(
            self, phase: co.Phase, time: float) -> float:
        r"""
        Compute the tendency of the displacement of the coupled oscillator.

        .. math::
            \dot{ξ} = \frac{p_ξ}{m}

        Parameters
        ----------
        phase : co.Phase
            The phase :math:`\boldsymbol π` of the coupled oscillator.
        time : float
            The current time :math:`t`.
        
        Returns
        -------
        float
            The tendency of the displacement :math:`\dot{ξ}`.
        """
        # get the oscillator properties
        m = self.osc_prop.mass(time)

        # get the phase properties
        p_ξ = phase.displacement_momentum

        # compute tendency
        dξdt = p_ξ / m
        return dξdt

    def tendency_angle_momentum(
            self, phase: co.Phase, time: float) -> float:
        r"""
        Compute the tendency of the angle momentum of the coupled oscillator.

        .. math::
            \dot{p}_φ = - m g ℓ \sin(\phi) - ϵ m g ξ \sin(\phi)

        Parameters
        ----------
        phase : co.Phase
            The phase :math:`\boldsymbol π` of the coupled oscillator.
        time : float
            The current time :math:`t`.
        
        Returns
        -------
        float
            The tendency of the angle momentum :math:`\dot{p}_φ`.
        """
        # get the oscillator properties
        m = self.osc_prop.mass(time)
        g = self.osc_prop.gravity(time)
        ϵ = self.osc_prop.epsilon(time)
        ℓ = self.osc_prop.length

        # get the phase properties
        φ = phase.angle
        ξ = phase.displacement

        # compute uncoupled tendency
        uncoupled_tendency = - m * g * ℓ * np.sin(φ)
        # compute coupled tendency
        coupled_tendency = - m * g * ξ * np.sin(φ)
        # compute full tendency
        dpφdt = uncoupled_tendency + ϵ * coupled_tendency
        return dpφdt
        
    def tendency_displacement_momentum(
            self, phase: co.Phase, time: float) -> float:
        r"""
        Compute the tendency of the displacement momentum of the coupled oscillator.

        .. math::
            \dot{p}_ξ = - k ξ + ϵ m g \cos(\phi) + ϵ \frac{p_φ^2}{m (ℓ + ξ)^3}

        Parameters
        ----------
        phase : co.Phase
            The phase :math:`\boldsymbol π` of the coupled oscillator.
        time : float
            The current time :math:`t`.
        
        Returns
        -------
        float
            The tendency of the displacement momentum :math:`\dot{p}_ξ`.
        """
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
        coupled_tendency = m * g * np.cos(φ) + p_φ**2 / (m * (ℓ + ξ)**3)
        # compute full tendency
        dpξdt = uncoupled_tendency + ϵ * coupled_tendency
        return dpξdt

    def tendency_phase(
            self, phase: co.Phase, time: float) -> co.Phase:
        r"""
        Compute the tendency of the phase of the coupled oscillator.

        .. math::
            \dot{\boldsymbol \pi} = \frac{d}{d t} \boldsymbol π

        Parameters
        ----------
        phase : co.Phase
            The phase :math:`\boldsymbol π` of the coupled oscillator.
        time : float
            The current time :math:`t`.

        Returns
        -------
        co.Phase
            The tendency of the phase :math:`\dot{\boldsymbol π}`.
        """
        return co.Phase(
            osc_prop=self.osc_prop,
            angle=self.tendency_angle(phase, time),
            displacement=self.tendency_displacement(phase, time),
            angle_momentum=self.tendency_angle_momentum(phase, time),
            displacement_momentum=self.tendency_displacement_momentum(phase, time),
        )

    # ================================================================
    #  Properties
    # ================================================================

    @property
    def osc_prop(self) -> co.OscillatorProperties:
        """The oscillator properties of the coupled oscillator."""
        return self._osc_prop