# importing libraries
from typing import Callable, Iterable
import numpy as np
import coupled_oscillator as co
from manimlib import *

RAMP_PERIOD = 10

class Ramping(InteractiveScene):
    direction = -1
    def construct(self):
        self.camera.background_color = WHITE

        axes = Axes(
            x_range=[0, RAMP_PERIOD, RAMP_PERIOD/10],
            y_range=[0, 1, 0.5],
            width=10,
            height=2.5,
            axis_config={"color": GREY},
        )
        axes.y_axis.add_numbers(num_decimal_places=1).set_color(GREY_E)

        title_bottom = Text("Coupling Coefficient").set_color(GREY_E).to_edge(UP).scale(0.8)
        axes.next_to(title_bottom, DOWN)

        self.add(axes, title_bottom)

        time_label = Text("Time").set_color(GREY_E).scale(0.6)
        time_label.next_to(axes, DOWN)
        self.add(time_label)

        t = np.linspace(0, RAMP_PERIOD, 100)[::self.direction]
        epsilon_func = co.ramp_functions.Exponential(
            start_time=0, period=RAMP_PERIOD, start_value=0, end_value=1)
        epsilon = epsilon_func(t)

        epsilon_points = np.array([t, epsilon]).T
        epsilon_graph = VMobject().set_points_as_corners(
                axes.c2p(*epsilon_points.T))
        epsilon_graph.set_color(BLACK)
        self.add(epsilon_graph)

        # create an invisible copy of the graph
        invisible_graph = epsilon_graph.copy().set_stroke(opacity=0)

        epsilon_dot = Dot(radius=0.1).move_to(epsilon_graph.get_start()).set_fill(BLACK)
        def update_epsilon_dot(m):
            current_pos = invisible_graph.get_end()
            time = axes.p2c(current_pos)[0]
            m.move_to(axes.c2p(time, epsilon_func(time)))
        epsilon_dot.add_updater(update_epsilon_dot)
        self.add(epsilon_dot)

        self.play(ShowCreation(invisible_graph), run_time=RAMP_PERIOD, rate_func=linear)
            
class Forwards(Ramping):
    direction = 1

class Backwards(Ramping):
    direction = -1
