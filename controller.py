import math
import glm
import numpy as np
from scipy.integrate import ode

GAIN = -20
SET_POINT = 0.07


class Controller:
    def __init__(self, agent):
        self.agent = agent
        self.app = agent.app
        self.name = "vs"
        self.gain = GAIN
        self.set_point = SET_POINT
        self.delay = 2
        self.focal_length = self.app.WIN_SIZE[1] / (2 * math.tan(math.radians(self.app.camera.fov / 2)))
        self.distance = 11.8
        self.output = (0, 0, 0, 2.4525)
        self.box_center = glm.vec4(0, 0, 1, 1)
        self.img_box_center = [0, 0, 0, 0]
        self.center_velocity = [0, 0]
        self.divergence = 0
        self.divergence_sample = []
        self.distances_over_time = []
        self.velocities_over_time = []
        self.pos_over_time = []
        self.divergence_over_time = []
        self.delta_time_over_time = []
        self.time_over_time = []
        self.set_point_over_time = []
        self.gain_over_time = []
        self.state_input_over_time = []
        self.theta_over_time = []
        self.theta_sp_over_time = []
        self.divergence_variance = 0
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
            self.name = f"stabilization_{self.gain}_{self.set_point}_{self.delay}"
        if self.n_iteration < self.delay:
            self.vertex_projection()
            self.get_distance()
            self.save_data()
            self.n_iteration += 1
        else:
            self.vertex_projection()
            self.centering()
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

    def centering(self):
        mass = self.agent.mass
        x, y, z, theta, psi, phi, vx, vy, vz, q, r, p = self.agent.states
        Ixx = self.agent.Ixx
        Iyy = self.agent.Iyy
        Izz = self.agent.Izz
        distance = self.distances_over_time[self.n_iteration - self.delay]
        velocity = self.velocities_over_time[self.n_iteration - self.delay]
        self.divergence = velocity / distance
        error = 0.25 - self.divergence
        self.r.set_f_params(0.25 - self.divergence)
        error_integral = self.r.integrate(self.r.t + self.app.delta_time / 1000)
        error_diff = (self.divergence_over_time[self.n_iteration - self.delay + 1] -
                      self.divergence_over_time[self.n_iteration - self.delay])/(self.app.delta_time / 1000)
        self.get_distance()
        mu_z = 20 * error + 0 * error_integral - 0 * error_diff + vz

        # Outer loop
        thrust = mass * math.sqrt(mu_z**2 + (9.81)**2)
        self.theta_sp = math.atan(mu_z/9.81)

        # Inner loop
        # thrust = (9.81 + self.k_y_d * (self.y_d_sp - vy) + self.k_y_p * (self.y_sp - y)) * mass / (np.cos(phi) * np.cos(theta))
        t_theta = (self.k_theta_d * (self.theta_d_sp - q) + self.k_theta_p * (self.theta_sp - theta)) * Ixx
        t_psi = (self.k_psi_d * (self.psi_d_sp - r) + self.k_psi_p * (self.psi_sp - psi)) * Iyy
        t_phi = (self.k_phi_d * (self.phi_d_sp - p) + self.k_phi_p * (self.phi_sp - phi)) * Izz
        self.output = np.array([t_theta, t_psi, t_phi, thrust])

    def stabilization(self):
        mass = self.agent.mass
        x, y, z, theta, psi, phi, vx, vy, vz, q, r, p = self.agent.states
        Ixx = self.agent.Ixx
        Iyy = self.agent.Iyy
        Izz = self.agent.Izz
        thrust = (9.81 + self.k_y_d*(self.y_d_sp - vy) + self.k_y_p*(self.y_sp - y)) * mass/(np.cos(phi)*np.cos(theta))
        t_theta = (self.k_theta_d*(self.theta_d_sp - q) + self.k_theta_p*(self.theta_sp - theta)) * Ixx
        t_psi = (self.k_psi_d*(self.psi_d_sp - r) + self.k_psi_p*(self.psi_sp - psi)) * Iyy
        t_phi = (self.k_phi_d*(self.phi_d_sp - p) + self.k_phi_p*(self.phi_sp - self.center_velocity)) * Izz
        self.output = np.array([t_theta, t_psi, t_phi, thrust])


    def constant_r(self):
        if self.n_iteration == 0:
            self.name = f"constant_r_{self.gain}_{self.set_point}_{self.delay}"
        if self.n_iteration < self.delay:
            self.get_distance()
            self.save_data()
            self.n_iteration += 1
        else:
            distance = self.distances_over_time[self.n_iteration - self.delay]
            velocity = self.velocities_over_time[self.n_iteration - self.delay]
            self.divergence = velocity / distance
            self.output = np.array([0, 0, 0, 0])
            self.set_point = self.set_point * 1.005
            # if self.n_iteration == 150:
            #     self.set_point = 0.2
            # if self.n_iteration == 300:
            #     self.set_point = 0.05
            # if self.n_iteration == 450:
            #     self.set_point = 0.2
            self.get_distance()
            self.divergence_sample.append(self.divergence)
            # if len(self.divergence_sample) >= 30:
            #     self.divergence_variance = np.var(self.divergence_sample)
            #     self.divergence_sample = []
            #     if self.divergence_variance > 0.001:
            #         self.set_point = self.set_point * 1.2
            #         self.gain = self.gain / 2
            self.save_data()
            self.n_iteration += 1

    def constant_r_dot(self):
        if self.n_iteration == 0:
            self.name = "r_dot"
        if self.n_iteration < self.delay:
            # save data
            self.get_distance()
            self.save_data()
            self.n_iteration += 1
        else:
            distance0 = self.distances_over_time[self.n_iteration-self.delay]
            distance1 = self.distances_over_time[self.n_iteration-self.delay+1]
            velocity0 = self.velocities_over_time[self.n_iteration-self.delay]
            velocity1 = self.velocities_over_time[self.n_iteration-self.delay+1]
            r0 = velocity0 / distance0
            r1 = velocity1 / distance1
            m_r_dot = (r1 - r0) / (self.app.delta_time / 1000)
            s_r_dot = self.set_point
            u = (s_r_dot - m_r_dot) * self.gain
            self.r.set_f_params(u)
            self.output = np.real(self.r.integrate(self.r.t + self.app.delta_time / 1000)) * np.array([0, 0, 1, 0, 0, 0])
            self.get_distance()
            self.save_data()
            self.n_iteration += 1

    def constant_tau_dot(self):
        if self.n_iteration == 0:
            self.name = "tau_dot"
        if self.n_iteration < 2:
            self.get_distance()
            self.save_data()
            self.n_iteration += 1
        else:
            distance0 = self.distances_over_time[self.n_iteration-2]
            distance1 = self.distances_over_time[self.n_iteration-1]
            velocity0 = self.velocities_over_time[self.n_iteration-2]
            velocity1 = self.velocities_over_time[self.n_iteration-1]
            tau0 = distance0 / velocity0
            tau1 = distance1 / velocity1
            m_tau_dot = (tau1 - tau0) / self.app.delta_time
            s_tau_dot = 0.4
            self.output += (self.set_point + m_tau_dot) * self.gain * glm.vec3(0, 0, 1)
            self.get_distance()
            self.save_data()
            self.n_iteration += 1

    def get_distance(self):
        x0, y0, z0 = self.app.scene.objects[0].states[0:3]
        x1, y1, z1 = self.app.scene.objects[1].states[0:3]
        self.distance = math.sqrt((z1 - z0)**2) - 1.2

    def save_data(self):
        self.distances_over_time.append(self.distance)
        self.velocities_over_time.append(self.agent.states[8])
        self.pos_over_time.append(self.agent.states[2])
        self.theta_over_time.append(self.agent.states[3])
        self.theta_sp_over_time.append(self.theta_sp)
        self.divergence_over_time.append(self.divergence)
        self.time_over_time.append(self.app.time)
        self.set_point_over_time.append(self.set_point)
        self.gain_over_time.append(self.gain)
        self.state_input_over_time.append(self.output[2])

    def on_init(self):
        # set up the ode:
        self.r = ode(TauController.f_zoh).set_integrator('lsoda', method='bdf')
        self.r.set_initial_value(0, self.app.time)