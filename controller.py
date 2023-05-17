import math
import glm
import numpy as np
from scipy.integrate import ode

DESIRED_F1 = np.array([-200, 200])  # deg
DESIRED_F2 = np.array([200, 200])
Z = 6
GAIN = -20
SET_POINT = 0.05


class Controller:
    def __init__(self, agent):
        self.agent = agent
        self.app = agent.app
        self.name = "vs"
        self.gain = GAIN
        self.set_point = SET_POINT
        self.delay = 10
        self.focal_length = self.app.WIN_SIZE[1] / (2 * math.tan(math.radians(self.app.camera.fov / 2)))
        self.distance = 11.8
        self.output = (0, 0, 0, 0, 0, 0)
        self.control_variable = 0
        self.divergence_sample = []
        self.distances_over_time = []
        self.velocities_over_time = []
        self.pos_over_time = []
        self.c_var_over_time = []
        self.delta_time_over_time = []
        self.time_over_time = []
        self.set_point_over_time = []
        self.gain_over_time = []
        self.state_input_over_time = []
        self.divergence_variance = 0
        self.vertex1 = glm.vec4(1, 1, 1, 1)
        self.vertex2 = glm.vec4(-1, 1, 1, 1)
        self.desired_features = np.hstack((DESIRED_F1, DESIRED_F2))
        self.img_features = [0, 0]
        self.img_jacobian = np.zeros((4, 6))
        self.feature1_over_time = []
        self.feature2_over_time = []
        self.n_iteration = 0


class VSController(Controller):
    def __init__(self, app):
        super().__init__(app)

    def control(self):
        self.vertex_projection()
        self.update_img_jacobian()
        self.compute_drone_velocity()

    def vertex_projection(self):
        m_cube_model = self.app.scene.objects[0].m_model
        mvp = np.matmul(np.matmul(self.agent.m_drone_proj, self.agent.m_drone_view), m_cube_model)
        clip_feature_1 = np.matmul(mvp, self.vertex1)
        clip_feature_2 = np.matmul(mvp, self.vertex2)
        ndc_feature_1 = clip_feature_1 / clip_feature_1[3]
        ndc_feature_2 = clip_feature_2 / clip_feature_2[3]
        self.img_features[0] = ndc_feature_1 * np.array([self.app.WIN_SIZE[0]/2, self.app.WIN_SIZE[1]/2, 1, 1])
        self.img_features[1] = ndc_feature_2 * np.array([self.app.WIN_SIZE[0]/2, self.app.WIN_SIZE[1]/2, 1, 1])

    def update_img_jacobian(self):
        for i, feature in enumerate(self.img_features):
            u = -feature[0]
            v = feature[1]
            self.img_jacobian[0 + i*2, 0] = - self.focal_length / self.distance
            self.img_jacobian[0 + i*2, 1] = 0
            self.img_jacobian[0 + i*2, 2] = u / self.distance
            self.img_jacobian[0 + i*2, 3] = u * v / self.focal_length
            self.img_jacobian[0 + i*2, 4] = - (self.focal_length**2 + u**2) / self.focal_length
            self.img_jacobian[0 + i*2, 5] = v
            self.img_jacobian[1 + i*2, 0] = 0
            self.img_jacobian[1 + i*2, 1] = - self.focal_length / self.distance
            self.img_jacobian[1 + i*2, 2] = v / self.distance
            self.img_jacobian[1 + i*2, 3] = (self.focal_length ** 2 + v ** 2) / self.focal_length
            self.img_jacobian[1 + i*2, 4] = -u * v / self.focal_length
            self.img_jacobian[1 + i*2, 5] = -u

    def compute_drone_velocity(self):
        # velocity_vector = np.array([0, 0, 5, 0, 0, 0])
        # output = np.matmul(self.img_jacobian, np.array([0, 0, 5, 0, 0, 0]))
        u1, v1 = -self.img_features[0][0], self.img_features[0][1]
        u2, v2 = -self.img_features[1][0], self.img_features[1][1]
        self.output += GAIN * (np.linalg.pinv(self.img_jacobian) @ np.transpose(self.desired_features - np.hstack([[u1, v1, u2, v2]])))


class TauController(Controller):
    def __init__(self, app):
        super().__init__(app)
        self.on_init()

    @staticmethod
    def f_zoh(t, x, arg1):
        u = arg1

        return [u]

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
            self.control_variable = velocity / distance
            self.output = (self.set_point - self.control_variable) * self.gain * np.array([0, 0, 0, 0, 0, 0])
            self.set_point = self.set_point * 1.005
            # if self.n_iteration == 150:
            #     self.set_point = 0.2
            # if self.n_iteration == 300:
            #     self.set_point = 0.05
            # if self.n_iteration == 450:
            #     self.set_point = 0.2
            self.get_distance()
            self.divergence_sample.append(self.control_variable)
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
        self.distance = math.sqrt((x1 - x0)**2 + (y1 - y0)**2 + (z1 - z0)**2) - 1.2

    def save_data(self):
        self.distances_over_time.append(self.distance)
        self.velocities_over_time.append(self.agent.states[8])
        self.pos_over_time.append(self.agent.states[2])
        self.c_var_over_time.append(self.control_variable)
        self.time_over_time.append(self.app.time)
        self.set_point_over_time.append(self.set_point)
        self.gain_over_time.append(self.gain)
        self.state_input_over_time.append(self.output[2])

    def on_init(self):
        # set up the ode:
        self.r = ode(TauController.f_zoh).set_integrator('lsoda', method='bdf')
        self.r.set_initial_value(0, self.app.time)