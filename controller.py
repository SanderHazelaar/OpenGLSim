from logger import *
import math
import glm
import numpy as np
from scipy.integrate import ode

GAIN = 20
SET_POINT = 0.1


class Controller:
    def __init__(self, agent):
        self.agent = agent
        self.app = agent.app
        self.name = "vs"
        self.gain = GAIN
        self.set_point = SET_POINT
        self.delay = 6
        self.distance = 11.8
        self.output = np.array([0, 0, 0, 2.4525])
        self.box_center = glm.vec4(0, 0, 1, 1)
        self.img_box_center_x = 0
        self.img_box_center_y = 0
        self.center_velocity = [0, 0]
        self.divergence = 0.1
        self.distance_est = 0
        self.n_iteration = 0
        self.adapt_count = 0
        self.distance_confidence = 0

        # Stabilisation controller gains and set points
        self.k_y_p = -0.02
        self.k_y_d = 1
        self.k_phi_p = 270
        self.k_phi_d = 25
        self.k_theta_p = 270
        self.k_theta_d = 35
        self.k_psi_p = 270
        self.k_psi_d = 25
        self.y_sp = 0
        self.y_d_sp = 0
        self.phi_sp = 0
        self.phi_d_sp = 0
        self.theta_sp = 0
        self.theta_d_sp = 0
        self.psi_sp = 0
        self.psi_d_sp = 0


class TauController(Controller):
    def __init__(self, app):
        super().__init__(app)
        self.on_init()

    @staticmethod
    def f_zoh(t, x, arg1):
        u = arg1

        return [u]

    def control(self):
        if self.n_iteration == 0:
            self.name = f"servoing_{self.gain}_{self.set_point}_{self.delay}"
        self.vertex_projection()
        self.get_distance()
        if self.n_iteration >= self.delay:
            self.get_divergence()
            self.estimate_distance()
            if self.distance_est < 10 and self.adapt_count == 0 and self.distance_est != 0:
                self.gain = 13
                self.set_point = 0.15
                self.adapt_count = 1
                self.distance_confidence = 0
            if self.distance_est < 4 and self.adapt_count == 1 and self.distance_est != 0:
                self.gain = 10
                self.set_point = 0.2
                self.adapt_count = 2
                self.distance_confidence = 0
            self.outer_loop_control()
            self.inner_loop_control()
        self.app.logger.add_step()
        self.n_iteration += 1

    def vertex_projection(self):
        m_cube_model = self.app.scene.objects[0].m_model
        mvp = np.matmul(np.matmul(self.agent.m_drone_proj, self.agent.m_drone_view), m_cube_model)
        clip_box_center = np.matmul(mvp, self.box_center)
        ndc_box_center = clip_box_center / clip_box_center[3]
        new_img_box_center = ndc_box_center * np.array([self.app.WIN_SIZE[0]/2, self.app.WIN_SIZE[1]/2, 1, 1])
        x = new_img_box_center[0]
        y = new_img_box_center[1]
        # self.img_box_center_x = x * math.cos(-self.agent.states[5]) - y * math.sin(-self.agent.states[5])
        # self.img_box_center_y = y * math.cos(-self.agent.states[5]) + x * math.sin(-self.agent.states[5])
        # self.img_box_center_x = x * math.cos(self.agent.states[5]) + y * math.sin(self.agent.states[5])
        # self.img_box_center_y = -x * math.sin(self.agent.states[5]) * math.cos(self.agent.states[3]) + y * math.cos(self.agent.states[5]) * math.cos(self.agent.states[3]) - 965.028 * math.sin(self.agent.states[3])
        rx = self.agent.states[3]
        rz = -self.agent.states[5]
        z = 965.028
        xmatrix = np.array([[1, 0, 0],
                            [0, math.cos(rx), - math.sin(rx)],
                            [0, math.sin(rx), math.cos(rx)]])
        zmatrix = np.array([[math.cos(rz), - math.sin(rz), 0],
                            [math.sin(rz), math.cos(rz), 0],
                            [0, 0, 1]])
        img_box_center = np.matmul(np.matmul(zmatrix, xmatrix), np.array([x, y, z]))
        a = 0
    def outer_loop_control(self):
        x, y, z, theta, psi, phi, vx, vy, vz, q, r, p = self.agent.states
        divergence_error = self.set_point - self.divergence
        self.r.set_f_params(self.set_point - self.divergence)
        error_integral = self.r.integrate(self.r.t + self.app.delta_time / 1000)
        mu_z = self.gain * divergence_error + 0 * error_integral + vz
        mu_x = 0.005 * self.img_box_center_x + 0.005 * self.app.logger.img_box_center_x_d[self.n_iteration - self.delay]
        mu_y = 0.1 * y
        thrust = self.agent.mass * math.sqrt(mu_z**2 + mu_x**2 + (9.81 - mu_y)**2)
        self.theta_sp = math.atan(mu_z/(9.81-mu_y))
        self.phi_sp = math.asin(self.agent.mass * (mu_x/thrust))
        self.output[3] = thrust

    def inner_loop_control(self):
        x, y, z, theta, psi, phi, vx, vy, vz, q, r, p = self.agent.states
        Ixx = self.agent.Ixx
        Iyy = self.agent.Iyy
        Izz = self.agent.Izz
        # thrust = (9.81 + self.k_y_d * (self.y_d_sp - vy) + self.k_y_p * (self.y_sp - y)) * mass / (np.cos(phi) * np.cos(theta))
        t_theta = (self.k_theta_d * (self.theta_d_sp - q) + self.k_theta_p * (self.theta_sp - theta)) * Ixx
        t_psi = (self.k_psi_d * (self.psi_d_sp - r) + self.k_psi_p * (self.psi_sp - psi)) * Iyy
        t_phi = (self.k_phi_d * (self.phi_d_sp - p) + self.k_phi_p * (self.phi_sp - phi)) * Izz
        self.output[0:3] = [t_theta, t_psi, t_phi]

    def get_distance(self):
        x0, y0, z0 = self.app.scene.objects[0].states[0:3]
        x1, y1, z1 = self.app.scene.objects[1].states[0:3]
        self.distance = math.sqrt((z1 - z0)**2) - 1.2

    def get_divergence(self):
        distance = self.app.logger.distance[self.n_iteration - self.delay]
        velocity = self.app.logger.state[self.n_iteration - self.delay][8]
        self.divergence = velocity / distance # + np.random.rand() * 0.01

    def estimate_distance(self):
        # if -0.01 < self.app.logger.divergence_d[self.n_iteration - self.delay] < 0.01 and self.divergence != 0:
        if -0.02 < self.divergence - self.set_point < 0.02 and self.divergence != 0:
            self.distance_confidence += 0.1
            self.distance_est = self.agent.states[3] * 17.2 * self.divergence**(-0.808)
            # self.distance_est = 0.7 * (math.sin(self.agent.states[3]) * self.output[3]) / \
            #                     (self.divergence**2 + self.app.logger.divergence_d[self.n_iteration - self.delay])
        if self.distance_confidence < 4:
            self.distance_est = 0

    def adapt_controller(self):
        self.gain = 10
        self.set_point = 0.2

    def on_init(self):
        # set up the ode:
        self.r = ode(TauController.f_zoh).set_integrator('lsoda', method='bdf')
        self.r.set_initial_value(0, self.app.time)
