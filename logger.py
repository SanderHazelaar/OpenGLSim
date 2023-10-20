import numpy as np
import os
class Logger:
    def __init__(self, app):
        self.app = app
        self.controller = app.scene.objects[1].controller
        self.state = []
        self.distance = []
        self.distance_est = []
        self.time = []
        self.divergence = []
        self.divergence_d = []
        self.img_box_center_x = []
        self.img_box_center_y = []
        self.img_box_center_x_d = []
        self.img_box_center_y_d = []

    def add_step(self):
        self.state.append(self.controller.agent.states)
        self.distance.append(self.controller.distance)
        self.distance_est.append(self.controller.distance_est)
        self.time.append(self.app.time)
        self.divergence.append(self.controller.divergence)
        self.divergence_d.append(self.derivative(self.divergence))
        self.img_box_center_x.append(self.controller.img_box_center_x)
        self.img_box_center_y.append(self.controller.img_box_center_y)
        self.img_box_center_x_d.append(self.derivative(self.img_box_center_x))
        self.img_box_center_y_d.append(self.derivative(self.img_box_center_y))

    def derivative(self, series):
        current_step = self.controller.n_iteration - self.controller.delay + 1
        if self.controller.n_iteration >= self.controller.delay:
            dxdt = (series[current_step] - series[current_step-1]) / (self.app.delta_time / 1000)
        else:
            dxdt = 0
        return dxdt

    def save_log_file(self):
        states = np.array(self.state)
        divergences = np.array(self.divergence)
        distances = np.array(self.distance)
        distance_est = np.array(self.distance_est)
        time = np.array(self.time) - self.time[0]
        data = np.block([[np.transpose(states)], [time], [divergences], [distances], [distance_est]])
        filename = os.path.join('sim_data', self.controller.name)
        np.save(filename, data)

