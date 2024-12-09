# importing libraries
from typing import Callable, Iterable
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import coupled_oscillator as co
from manimlib import *

RAMP_PERIOD = 10

def get_forward_evolution():
    # defining the parameters
    endtime = 4 * np.pi
    gravity = 1
    mass = 1
    length = 1
    spring_constant = 10**2  # freqency of spring is 10 times the frequency of pendulum
    osc_prop = co.OscillatorProperties(
        gravity=gravity, mass=mass, length=length, spring_constant=spring_constant, epsilon=1)
    hamiltonian = co.Hamiltonian(osc_prop)  # This defines the system of equations

    # defining the initial conditions
    angle_deg = -45
    angle_rad = np.deg2rad(angle_deg)
    phase_ini = co.Phase(osc_prop, angle=angle_rad)

    epsilon = co.ramp_functions.Exponential(
        start_time=0, period=RAMP_PERIOD, start_value=1, end_value=0)

    # create a new oscillator with the ramped down coupling
    osc_prop_ramp = co.OscillatorProperties(
        gravity=gravity, mass=mass, length=length, spring_constant=spring_constant,
        epsilon=epsilon)

    # create a new Hamiltonian with the ramped down coupling
    ramp_hamiltonian = co.Hamiltonian(osc_prop_ramp)

    # solve the system with the ramped down coupling
    forwards_evolution = co.solve(
        hamiltonian=ramp_hamiltonian, 
        phase=phase_ini,
        num_points=1000,
        time_span=[0, RAMP_PERIOD])

    return forwards_evolution, hamiltonian, ramp_hamiltonian, osc_prop

def get_backward_evolution():
    forwards_evolution, hamiltonian, ramp_hamiltonian, osc_prop = get_forward_evolution()
    # get the final phase in the uncoupled system
    uncoupled_phase = co.Phase.from_xr(osc_prop, forwards_evolution.isel(time=-1))
    # eliminate the displacement
    eliminate_displacement = co.balancing.NaiveBalancer(hamiltonian).balance
    uncoupled_bal_phase = eliminate_displacement(uncoupled_phase, time=0)
    # integrate the uncoupled system back to the start
    backwards_evolution = co.solve(
        hamiltonian=ramp_hamiltonian, 
        phase=uncoupled_bal_phase,
        num_points=1000,
        time_span=[RAMP_PERIOD, 0])

    return backwards_evolution

def evolution_to_points(evolution):
    x = evolution["time"].values
    y = - evolution["displacement"].values
    points = np.array([x, y]).T
    return points

class OptimalBalanceForward(InteractiveScene):
    def construct(self):
        self.camera.background_color = WHITE

        axes_top = Axes(
            x_range=[0, RAMP_PERIOD, RAMP_PERIOD/10],
            y_range=[-0.03, 0.01, 0.01],
            width=10,
            height=2.5,
            axis_config={"color": GREY},
        )

        title = Text("Displacement: ").set_color(GREY_E)
        title_math = Tex(r"\delta x").next_to(title, RIGHT).set_color(GREY_E).set_stroke(width=1.2)
        top_title = VGroup(title, title_math)
        top_title.to_edge(UP).scale(0.8).shift(UP*0.3)
        axes_top.next_to(top_title, DOWN)
        self.add(axes_top, top_title)

        axes_bottom = Axes(
            x_range=[0, RAMP_PERIOD, RAMP_PERIOD/10],
            y_range=[0, 1, 0.5],
            width=10,
            height=2.5,
            axis_config={"color": GREY},
        )
        axes_bottom.y_axis.add_numbers(num_decimal_places=1).set_color(GREY_E)

        title_bottom = Text("Coupling Coefficient").set_color(GREY_E).next_to(axes_top, DOWN).scale(0.8)
        axes_bottom.next_to(title_bottom, DOWN)

        self.add(axes_bottom, title_bottom)

        forwards_evolution, *_ = get_forward_evolution()
        forwards_points = evolution_to_points(forwards_evolution)

        forwards_graph = VMobject().set_points_as_corners(
                axes_top.c2p(*forwards_points.T))
        forwards_graph.set_color(BLUE_E)
        self.add(forwards_graph)

        time_label = Text("Time").set_color(GREY_E).scale(0.6)
        time_label.next_to(axes_bottom, DOWN)
        self.add(time_label)

        t = np.linspace(0, RAMP_PERIOD, 100)
        epsilon_func = co.ramp_functions.Exponential(
            start_time=0, period=RAMP_PERIOD, start_value=1, end_value=0)
        epsilon = epsilon_func(t)

        epsilon_points = np.array([t, epsilon]).T
        epsilon_graph = VMobject().set_points_as_corners(
                axes_bottom.c2p(*epsilon_points.T))
        epsilon_graph.set_color(BLACK)
        self.add(epsilon_graph)

        # add dots to the graph
        forward_dot = Dot(radius=0.1).move_to(forwards_graph.get_start()).set_fill(BLUE_E)
        forward_dot.add_updater(lambda m: m.move_to(forwards_graph.get_end()))
        self.add(forward_dot)

        epsilon_dot = Dot(radius=0.1).move_to(epsilon_graph.get_start()).set_fill(BLACK)
        def update_epsilon_dot(m):
            time = axes_top.p2c(forward_dot.get_center())[0]
            m.move_to(axes_bottom.c2p(time, epsilon_func(time)))
        epsilon_dot.add_updater(update_epsilon_dot)
        self.add(epsilon_dot)

        self.play(ShowCreation(forwards_graph), run_time=RAMP_PERIOD, rate_func=linear)
            

class OptimalBalanceBackward(InteractiveScene):
    def construct(self):
        self.camera.background_color = WHITE

        axes_top = Axes(
            x_range=[0, RAMP_PERIOD, RAMP_PERIOD/10],
            y_range=[-0.03, 0.01, 0.01],
            width=10,
            height=2.5,
            axis_config={"color": GREY},
        )

        title = Text("Displacement: ").set_color(GREY_E)
        title_math = Tex(r"\delta x").next_to(title, RIGHT).set_color(GREY_E).set_stroke(width=1.2)
        top_title = VGroup(title, title_math)
        top_title.to_edge(UP).scale(0.8).shift(UP*0.3)
        axes_top.next_to(top_title, DOWN)
        self.add(axes_top, top_title)

        axes_bottom = Axes(
            x_range=[0, RAMP_PERIOD, RAMP_PERIOD/10],
            y_range=[0, 1, 0.5],
            width=10,
            height=2.5,
            axis_config={"color": GREY},
        )
        axes_bottom.y_axis.add_numbers(num_decimal_places=1).set_color(GREY_E)

        title_bottom = Text("Coupling Coefficient").set_color(GREY_E).next_to(axes_top, DOWN).scale(0.8)
        axes_bottom.next_to(title_bottom, DOWN)

        self.add(axes_bottom, title_bottom)

        forwards_evolution, *_ = get_forward_evolution()
        forwards_points = evolution_to_points(forwards_evolution)

        forwards_graph = VMobject().set_points_as_corners(
                axes_top.c2p(*forwards_points.T))
        forwards_graph.set_color(BLUE_E)
        self.add(forwards_graph)

        time_label = Text("Time").set_color(GREY_E).scale(0.6)
        time_label.next_to(axes_bottom, DOWN)
        self.add(time_label)

        t = np.linspace(0, RAMP_PERIOD, 100)
        epsilon_func = co.ramp_functions.Exponential(
            start_time=0, period=RAMP_PERIOD, start_value=1, end_value=0)
        epsilon = epsilon_func(t)

        epsilon_points = np.array([t, epsilon]).T
        epsilon_graph = VMobject().set_points_as_corners(
                axes_bottom.c2p(*epsilon_points.T))
        epsilon_graph.set_color(BLACK)
        self.add(epsilon_graph)

        # backwards evolution
        backwards_evolution = get_backward_evolution()
        backwards_points = evolution_to_points(backwards_evolution)

        backwards_graph = VMobject().set_points_as_corners(
                axes_top.c2p(*backwards_points.T))
        backwards_graph.set_color(RED_E)
        self.add(backwards_graph)

        # add dots to the graph
        backward_dot = Dot(radius=0.1).move_to(forwards_graph.get_start()).set_fill(RED_E)
        backward_dot.add_updater(lambda m: m.move_to(backwards_graph.get_end()))
        self.add(backward_dot)

        epsilon_dot = Dot(radius=0.1).move_to(epsilon_graph.get_start()).set_fill(BLACK)
        def update_epsilon_dot(m):
            time = axes_top.p2c(backward_dot.get_center())[0]
            m.move_to(axes_bottom.c2p(time, epsilon_func(time)))
        epsilon_dot.add_updater(update_epsilon_dot)
        self.add(epsilon_dot)

        self.play(ShowCreation(backwards_graph), run_time=RAMP_PERIOD, rate_func=linear)
