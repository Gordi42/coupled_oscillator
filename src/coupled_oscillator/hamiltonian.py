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
            (\ell + \xi(t)) \sin(\phi(t)) \\ 
        -(\ell + \xi(t)) \cos(\phi(t)) 
        \end{matrix} \right)

    where :math:`\ell` is the rest length of the spring. 
    We transform our system from the cartesian coordinates 
    :math:`(x, y)` to the generalized coordinates :math:`(\phi, \xi)`.

    Forces and Potentials
    ~~~~~~~~~~~~~~~~~~~~~

    The spring force is given by:

    .. math::
        \boldsymbol F_{\text{spring}} 
        = -k \xi \frac{\boldsymbol r}{\| \boldsymbol r \|}

    where :math:`k(t)` is a time dependent spring constant. Using
    :math:`F = -\nabla V` we can find the potential of the spring:

    .. math::
        V_{\text{spring}}(\xi, t) = \frac{1}{2} k \xi^2

    In addition to the spring force, there is a gravitational force 
    acting on the mass. The gravitational potential is given by:

    .. math::
        V_{\text{gravity}}(\phi, \xi, t) = m g y = -m g (\ell + \xi) \cos(\phi)

    where :math:`m(t)` and :math:`g(t)` are time dependent mass and 
    gravitational acceleration, respectively.

    Kinetic Energy
    ~~~~~~~~~~~~~~
    To find the hamiltonian, we need to find the kinetic energy. 
    The kinetic energy is given by:

    .. math::
        T = \frac{1}{2} m \dot{\boldsymbol r}^2 
        = \frac{1}{2} m \left( \dot{\xi}^2 + (\ell + \xi)^2 \dot{\phi}^2 \right)

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
        p_\phi = \frac{\partial L}{\partial \dot{\phi}} = m (\ell + \xi)^2 \dot{\phi}

        p_\xi = \frac{\partial L}{\partial \dot{\xi}} = m \dot{\xi}

    Hamiltonian
    ~~~~~~~~~~~
    The hamiltonian of our system is given by the sum of the kinetic
    and potential energy:

    .. math::
        H(\boldsymbol \pi, t) = T + V = 
        \frac{p_\phi^2}{2 m (\ell + \xi)^2} + \frac{p_\xi^2}{2 m}
        + \frac{k \xi^2}{2} - m g (\ell + \xi) \cos(\phi)

    where :math:`\boldsymbol \pi = (\phi, \xi, p_\phi, p_\xi)` is the
    phase vector of the system.
    We rewrite the hamiltonian and sort it by coupled and uncoupled terms:

    .. math::
        H(\boldsymbol \pi, t) = H_0(\boldsymbol \pi, t) + \epsilon H_1(\boldsymbol \pi, t)

    where :math:`\epsilon` is a scaling factor that determines the strength of the coupling.
    For :math:`\epsilon = 0` we get the uncoupled hamiltonian and for :math:`\epsilon = 1`
    we get the fully coupled hamiltonian. The uncoupled part is given by:

    .. math::
        H_0(\boldsymbol \pi, t) = 
        \frac{p_\phi^2}{2 m \ell^2} + \frac{p_\xi^2}{2 m}
        + \frac{k \xi^2}{2} - m g \ell \cos(\phi)

    The coupled part is given by:

    .. math::
        H_1(\boldsymbol \pi, t) = 
        - \frac{\xi (\xi + 2 \ell) p_\phi^2}{2 m \ell^2 (\ell + \xi)^2}
        - m g \xi \cos(\phi) 

    Due to the high nonlinearity of the coupled part, the equation becomes
    hard to analyze. It can be simplified by assuming that the spring displacement
    is small compared to the rest length of the spring. And that the angle is small.
    This approximated hamiltonian is detailed in [TODO].
  
    Hamilton's Equations
    --------------------

    The tendency equations are given by the time derivative of the phase vector:

    .. math::
        \dot{\boldsymbol \pi} = \frac{d}{d t} \boldsymbol \pi

    using the hamiltonian, we can write the tendency equations as:

    .. math::
        \dot{\phi} 
        = \frac{\partial H}{\partial p_\phi} 
        = \frac{p_\phi}{m \ell^2} - \epsilon \frac{\xi (\xi + 2 \ell)}{m \ell^2 (\ell + \xi)^2} p_\phi

        \dot{\xi} 
        = \frac{\partial H}{\partial p_\xi} 
        = \frac{p_\xi}{m}

        \dot{p_\phi} = - \frac{\partial H}{\partial \phi} 
            = - m g \ell \sin(\phi) - \epsilon m g \xi \sin(\phi)

        \dot{p_\xi} = - \frac{\partial H}{\partial \xi}
            = - k \xi + \epsilon m g \cos(\phi) + \epsilon \frac{p_\phi^2}{m (\ell + \xi)^3} 

    Uncoupled Solution
    ------------------
    We set :math:`\epsilon = 0` to obtain the uncoupled system, further 
    we assume small angles :math:`\phi`, such that

    .. math::
        \sin(\phi) \approx \phi 

    this will obtain the hamiltonian equations:

    .. math::
        \dot{\phi} = \frac{p_\phi}{m \ell^2}
        \quad , \quad
        \dot{\xi} = \frac{p_\xi}{m}
        \quad , \quad
        \dot{p_\phi} = -mg\ell \phi
        \quad , \quad
        \dot{p_\xi} = -k \xi 

    Combining them yields:

    .. math::
        \ddot{\phi} = -\omega_\phi^2 \phi
        \quad , \quad
        \ddot{\xi} = -\omega_\xi^2 \xi

    with frequencies:

    .. math::
        \omega_\phi^2 = \frac{g}{\ell}
        \quad , \quad
        \omega_\xi^2 = \frac{k}{m}

    these are the well known frequencies of the string pendulum 
    and of the spring pendulum.

    """
    def __init__(self, osc_prop: co.OscillatorProperties) -> None:
        self._osc_prop = osc_prop

    def kinetic_energy(self, phase: co.Phase, time: float) -> float:
        ...

    def potential_energy(self, phase: co.Phase, time: float) -> float:
        ...

    def total_energy(self, phase: co.Phase, time: float) -> float:
        ekin = self.kinetic_energy(phase, time)
        epot = self.potential_energy(phase, time)
        etot = ekin + epot
        return etot

    def tendency_angle(
            self, phase: co.Phase, time: float) -> float:
        ...
    
    def tendency_displacement(
            self, phase: co.Phase, time: float) -> float:
        ...

    def tendency_angle_momentum(
            self, phase: co.Phase, time: float) -> float:
        ...

    def tendency_displacement_momentum(
            self, phase: co.Phase, time: float) -> float:
        ...

    def tendency_phase(
            self, phase: co.Phase, time: float) -> co.Phase:
        return co.Phase(
            osc_prop=self._osc_prop,
            angle=self.tendency_angle(phase, time),
            displacement=self.tendency_displacement(phase, time),
            angle_momentum=self.tendency_angle_momentum(phase, time),
            displacement_momentum=self.tendency_displacement_momentum(phase, time),
        )