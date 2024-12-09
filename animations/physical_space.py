# importing libraries
from typing import Callable, Iterable
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import coupled_oscillator as co
from manimlib import *

RUN_TIME = 30


def create_hamiltonian():
    # defining the parameters
    gravity = 1
    mass = 1
    length = 3
    spring_constant = (10)**2  # freqency of spring is 10 times the frequency of pendulum
    osc_prop = co.OscillatorProperties(
        gravity=gravity, mass=mass, length=length, spring_constant=spring_constant)
    hamiltonian = co.Hamiltonian(osc_prop)  # This defines the system of equations
    return hamiltonian

def get_solution(ini_disp=0, balance=False):
    hamiltonian = create_hamiltonian()
    osc_prop = hamiltonian.osc_prop

    # defining the initial conditions
    angle_deg = -35
    angle_rad = np.deg2rad(angle_deg)
    spring_displacement = ini_disp
    phase_ini = co.Phase(osc_prop, angle=angle_rad, displacement=spring_displacement)

    # balance the initial conditions
    if balance:
        optimal_balance = co.balancing.OptimalBalance(
            hamiltonian=hamiltonian,
            linear_balance=co.balancing.NaiveBalancer(hamiltonian),
            ramp_method=co.ramp_functions.Exponential,
            ramp_period=10, 
            max_it=5,
            stop_criterion=1e-12)
        phase_ini = optimal_balance.balance(phase_ini, time=0)

    # Let's solve the system for one period of the pendulum
    solution = co.solve(
        hamiltonian=hamiltonian, 
        phase=phase_ini, 
        time_span=[0, RUN_TIME],
        num_points=1000)
    return solution

def get_configuration_evolution(ini_disp=0, balance=False):
    hamiltonian = create_hamiltonian()
    osc_prop = hamiltonian.osc_prop
    solution = get_solution(ini_disp, balance)

    points = np.array([*zip(solution["angle"].values, 
                            -solution["displacement"].values, 
                            np.zeros_like(solution["angle"].values))])
    return points

def get_cartesian_evolution(ini_disp=0, balance=False):
    hamiltonian = create_hamiltonian()
    osc_prop = hamiltonian.osc_prop
    solution = get_solution(ini_disp, balance)

    def generalized_to_cartesian(ds: xr.Dataset) -> xr.Dataset:
        """Converts the generalized coordinates to cartesian coordinates"""
        # get the phase object from the dataset
        phase = co.Phase.from_xr(osc_prop=osc_prop, xr_data=ds)
        # use the phase object to convert the generalized coordinates to cartesian coordinates
        coords = phase.to_cartesian_coordinates()
        # create a new dataset with the cartesian coordinates
        ds_cartesian = xr.Dataset({
            "time": xr.DataArray([ds["time"].item()], dims="time"),
            "x": xr.DataArray([coords[0]], dims="time"),
            "y": xr.DataArray([coords[1]], dims="time"),})
        return ds_cartesian

    # apply the transformation to the solution
    cart_solution = solution.groupby("time").map(generalized_to_cartesian)

    # extract the x and y coordinates
    x = cart_solution["x"].values
    y = cart_solution["y"].values

    # convert to points
    points = np.array([*zip(x, y, np.zeros_like(x))])

    return points


def setup_physical_space():
    # Create a 2D axes
    axes = Axes(
        x_range=[-2, 2, 4],
        y_range=[-4, 0, 4],
        width=6,
        height=6,
        axis_config={"color": GREY},
    )
    return axes

def create_pendulum(trajectory, physical_axes):
    # Create a sphere for the mass
    sphere = Sphere(radius=0.2, resolution=(100, 100), color=BLUE_E)
    sphere.add_updater(lambda m: m.move_to(trajectory.get_end()))

    # Create a tail for the sphere
    tail = TracingTail(sphere, stroke_width=4, stroke_color=BLUE_E, time_traced=1)

    # Create a line for the pendulum
    pivot = physical_axes.c2p(0, 0)
    pendulum_line = Line(pivot, pivot, color=GREY_E)
    pendulum_line.add_updater(
            lambda m: m.put_start_and_end_on(pivot, trajectory.get_end()))

    return Group(tail, pendulum_line, sphere)

def create_pendulum_labels(trajectory, physical_axes):
    axes = physical_axes

    pivot = axes.c2p(0, 0)

    # define a helper function to get the angle of the pendulum
    def get_angle(point):
        # first we need to transform the point back to the original coordinates
        coords = axes.p2c(point)
        x, y = coords
        return - np.arctan2(-x, -y)

    # add a arc to represent the angle
    arc = Arc(radius=2.0, start_angle=-PI/2, angle=0.2, color=RED_E, arc_center=pivot)

    def update_arc(arc):
        new_angle = get_angle(trajectory.get_end())
        new_arc = Arc(radius=2.0, start_angle=-PI/2, 
                      angle=new_angle, color=RED_E, arc_center=pivot)
        arc.become(new_arc)
    
    arc.add_updater(update_arc)

    # add a label \theta below the arc
    theta_label = Tex(r"\theta")
    theta_label.set_color(RED_E)

    def position_theta_label(m):
        new_angle = get_angle(trajectory.get_end()) / 2
        distance = 1.1
        x = np.sin(new_angle) * distance
        y = -np.cos(new_angle) * distance
        m.move_to(axes.c2p(x, y))
    
    # theta_label.add_updater(lambda m: m.next_to(arc, UP, buff=0.1))
    theta_label.add_updater(position_theta_label)

    angle_label = Group(arc, theta_label)


    # add a label L + \delta x to the right of the pendulum line
    length_label = Tex(r"L + \delta x")
    length_label.rotation = PI/2
    length_label.set_color(GREY_E)

    def position_label(label):
        # get the angle of the pendulum
        new_angle = get_angle(trajectory.get_end())

        # move the label down a bit
        distance = 2
        x = np.sin(new_angle) * distance
        y = -np.cos(new_angle) * distance
        label.move_to(axes.c2p(x, y) + RIGHT * 0.4)
        
        # rotate the label to be parallel to the pendulum line
        angle = get_angle(trajectory.get_end()) - label.rotation
        label.rotation += angle
        label.rotate(angle, about_point=label.get_center())

    length_label.add_updater(position_label)

    return Group(angle_label, length_label)

def create_phase_space(scale=1):
    axes = Axes(
        x_range=[-PI/4, PI/4, PI/8],
        y_range=[-0.05, 0.05, 0.05],
        width=10*scale,
        height=6*scale,
        axis_config={"color": GREY_E},
    )
    labels = axes.get_axis_labels(x_label_tex=r"\theta", y_label_tex=r"\delta x")
    labels.set_stroke(GREY_E, 1)
    labels.set_fill(GREY_E, 1)
    return axes, labels

def create_phase_space_trajectory(axes, trajectory):
    # create a dot at the end of the curve
    dot = Dot(axes.c2p(0,0), radius=0.15)
    dot.set_fill(BLUE_E)
    dot.add_updater(lambda m: m.move_to(trajectory.get_end()))

    tail = TracingTail(dot, stroke_width=4, stroke_color=BLUE_E, time_traced=10)
    return Group(tail, dot)


class PhysicalSpacePart1(InteractiveScene):
    def construct(self):
        self.camera.background_color = WHITE

        axes = setup_physical_space()
        axes.shift(LEFT * 3.7)
        self.add(axes)


        points = get_cartesian_evolution(ini_disp=-0.1)

        curve = VMobject().set_points_as_corners(axes.c2p(*points.T))
        curve.set_stroke(opacity=0)

        pendulum = create_pendulum(curve, axes)
        pendulum_label = create_pendulum_labels(curve, axes)
        self.add(pendulum, pendulum_label)

        self.play(ShowCreation(curve, run_time=RUN_TIME, rate_func=linear))

class PhysicalSpacePart2(InteractiveScene):
    def construct(self):
        self.camera.background_color = WHITE

        axes = setup_physical_space()
        axes.shift(LEFT * 3.7)
        self.add(axes)


        points = get_cartesian_evolution(ini_disp=0)

        curve = VMobject().set_points_as_corners(axes.c2p(*points.T))
        curve.set_stroke(opacity=0)

        pendulum = create_pendulum(curve, axes)
        pendulum_label = create_pendulum_labels(curve, axes)
        self.add(pendulum, pendulum_label)

        self.play(ShowCreation(curve, run_time=RUN_TIME, rate_func=linear))


class ConfigurationSpace(InteractiveScene):
    def construct(self):
        self.camera.background_color = WHITE

        axes, labels = create_phase_space()
        self.add(axes, labels)

        # Add a title "Configuration Space" to the scene
        title = Text("Configuration Space")
        title.set_color(BLUE_E)
        title.next_to(axes, UP)
        self.add(title)

        # Get the solution
        points = get_configuration_evolution()
        curve = VMobject().set_points_as_corners(axes.c2p(*points.T))
        curve.set_stroke(BLUE_E, opacity=0)

        self.add(create_phase_space_trajectory(axes, curve))
        self.play(ShowCreation(curve, run_time=RUN_TIME, rate_func=linear))


class CombinedSpacePart1(InteractiveScene):
    def construct(self):
        self.camera.background_color = WHITE

        physical_axes = setup_physical_space()
        physical_axes.shift(LEFT * 3.7)
        phase_axes, phase_labels = create_phase_space(scale=0.67)
        phase_axes.shift(RIGHT * 3.5 + DOWN * 0.5)
        phase_labels.shift(RIGHT * 3.5 + DOWN * 0.5)

        self.add(physical_axes, phase_axes, phase_labels)

        cart_solution = get_cartesian_evolution()
        conf_solution = get_configuration_evolution()

        cart_traj = VMobject().set_points_as_corners(
                physical_axes.c2p(*cart_solution.T))
        cart_traj.set_stroke(opacity=0)

        conf_traj = VMobject().set_points_as_corners(
                phase_axes.c2p(*conf_solution.T))
        conf_traj.set_stroke(opacity=0)

        pendulum = create_pendulum(cart_traj, physical_axes)
        pendulum_label = create_pendulum_labels(cart_traj, physical_axes)
        self.add(pendulum, pendulum_label)

        self.add(create_phase_space_trajectory(phase_axes, conf_traj))
        # Add a title "Configuration Space" to the scene
        title = Text("Configuration Space")
        title.set_color(BLUE_E)
        title.next_to(phase_axes, UP)
        self.add(title)

        self.play(ShowCreation(cart_traj, run_time=RUN_TIME, rate_func=linear),
                  ShowCreation(conf_traj, run_time=RUN_TIME, rate_func=linear))


class CombinedSpacePart2(InteractiveScene):
    def construct(self):
        self.camera.background_color = WHITE

        physical_axes = setup_physical_space()
        physical_axes.shift(LEFT * 3.7)
        phase_axes, phase_labels = create_phase_space(scale=0.67)
        phase_axes.shift(RIGHT * 3.5 + DOWN * 0.5)
        phase_labels.shift(RIGHT * 3.5 + DOWN * 0.5)

        self.add(physical_axes, phase_axes, phase_labels)

        cart_solution = get_cartesian_evolution(balance=True)
        conf_solution = get_configuration_evolution(balance=True)

        cart_traj = VMobject().set_points_as_corners(
                physical_axes.c2p(*cart_solution.T))
        cart_traj.set_stroke(opacity=0)

        conf_traj = VMobject().set_points_as_corners(
                phase_axes.c2p(*conf_solution.T))
        conf_traj.set_stroke(opacity=0)

        pendulum = create_pendulum(cart_traj, physical_axes)
        pendulum_label = create_pendulum_labels(cart_traj, physical_axes)
        self.add(pendulum, pendulum_label)

        self.add(create_phase_space_trajectory(phase_axes, conf_traj))
        # Add a title "Configuration Space" to the scene
        title = Text("Configuration Space")
        title.set_color(BLUE_E)
        title.next_to(phase_axes, UP)
        self.add(title)

        self.play(ShowCreation(cart_traj, run_time=RUN_TIME, rate_func=linear),
                  ShowCreation(conf_traj, run_time=RUN_TIME, rate_func=linear))

