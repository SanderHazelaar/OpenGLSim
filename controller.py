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
        self.delay = 2
        self.distance = 11.8
        self.output = np.array([0, 0, 0, 2.4525])
        self.box_center = glm.vec4(0, 0, 1, 1)
        self.img_box_center = [0, 0, 0, 0]
        self.center_velocity = [0, 0]
        self.divergence = 0
        self.divergence_var = 0
        self.distance_est = 15
        self.states_over_time = []
        self.distances_over_time = []
        self.divergence_over_time = []
        self.delta_time_over_time = []
        self.time_over_time = []
        self.gain_over_time = []
        self.state_input_over_time = []
        self.distance_est_over_time = []
        self.n_iteration = 0

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
            if self.divergence_var < 0.8:
                self.estimate_distance()
                if self.distance_est < 7:
                    self.adapt_controller()
            self.outer_loop_control()
            self.inner_loop_control()
        self.save_data()
        self.n_iteration += 1

    def vertex_projection(self):
        m_cube_model = self.app.scene.objects[0].m_model
        mvp = np.matmul(np.matmul(self.agent.m_drone_proj, self.agent.m_drone_view), m_cube_model)
        clip_box_center = np.matmul(mvp, self.box_center)
        ndc_box_center = clip_box_center / clip_box_center[3]
        new_img_box_center = ndc_box_center * np.array([self.app.WIN_SIZE[0]/2, self.app.WIN_SIZE[1]/2, 1, 1])
        self.center_velocity = (new_img_box_center - self.img_box_center) / (self.app.delta_time / 1000)
        self.img_box_center = new_img_box_center
        x = self.img_box_center[0]
        y = self.img_box_center[1]
        self.img_box_center[0] = x * math.cos(-self.agent.states[5]) - y * math.sin(-self.agent.states[5])
        self.img_box_center[1] = y * math.cos(-self.agent.states[5]) + x * math.sin(-self.agent.states[5])

    def outer_loop_control(self):
        x, y, z, theta, psi, phi, vx, vy, vz, q, r, p = self.agent.states
        divergence_error = self.set_point - self.divergence
        self.r.set_f_params(self.set_point - self.divergence)
        error_integral = self.r.integrate(self.r.t + self.app.delta_time / 1000)
        error_diff = (self.divergence_over_time[self.n_iteration - self.delay + 1] -
                      self.divergence_over_time[self.n_iteration - self.delay])/(self.app.delta_time / 1000)
        mu_z = self.gain * divergence_error + 0 * error_integral - 0 * error_diff + vz
        thrust = self.agent.mass * math.sqrt(mu_z**2 + (9.81)**2)
        self.theta_sp = math.atan(mu_z/9.81)
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
        distance = self.distances_over_time[self.n_iteration - self.delay]
        velocity = self.states_over_time[self.n_iteration - self.delay][8]
        self.divergence = velocity / distance

    def estimate_distance(self):
        self.distance_est = self.agent.states[3] * 17.2 * self.divergence**(-0.808)

    def adapt_controller(self):
        self.gain = self.gain / 1.5
        self.set_point = self.set_point * 2

    def save_data(self):
        self.states_over_time.append(self.agent.states)
        self.distances_over_time.append(self.distance)
        self.distance_est_over_time.append(self.distance_est)
        self.divergence_over_time.append(self.divergence)
        self.time_over_time.append(self.app.time)
        self.gain_over_time.append(self.gain)
        self.state_input_over_time.append(self.output[2])

    def on_init(self):
        # set up the ode:
        self.r = ode(TauController.f_zoh).set_integrator('lsoda', method='bdf')
        self.r.set_initial_value(0, self.app.time)