from controller import *
import moderngl as mgl
import numpy as np
import glm
from scipy.integrate import ode


class BaseModel:
    def __init__(self, app, vao_name, tex_id, pos=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1),
                 velocity=(0, 0, 0), ang_velocity=(0, 0, 0), mass=0.250, Ixx=0.005, Iyy=0.005, Izz=0.005,
                 drag_factor=0.25):
        self.app = app
        self.states = np.concatenate([pos, [np.radians(a) for a in rot], velocity, ang_velocity])
        self.scale = scale
        self.mass = mass
        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Izz = Izz
        self.drag_factor = drag_factor
        self.up = glm.vec3(0, 1, 0)
        self.right = glm.vec3(1, 0, 0)
        self.forward = glm.vec3(0, 0, 1)
        self.m_model = self.get_model_matrix()
        self.tex_id = tex_id
        self.vao = app.mesh.vao.vaos[vao_name]
        self.program = self.vao.program
        self.camera = self.app.camera
        self.controller = TauController(self)

    def update(self): ...

    def get_model_matrix(self):
        m_model = glm.mat4()
        # translate
        m_model = glm.translate(m_model, glm.vec3(list(self.states[0:3])))
        # rotate
        m_model = glm.rotate(m_model, self.states[5], glm.vec3(0, 0, 1))
        m_model = glm.rotate(m_model, self.states[4], glm.vec3(0, 1, 0))
        m_model = glm.rotate(m_model, self.states[3], glm.vec3(1, 0, 0))
        # scale
        m_model = glm.scale(m_model, self.scale)
        return m_model

    def render(self):
        self.update()
        self.vao.render()


class ExtendedBaseModel(BaseModel):
    def __init__(self, app, vao_name, tex_id, pos, rot, scale, velocity, ang_velocity):
        super().__init__(app, vao_name, tex_id, pos, rot, scale, velocity, ang_velocity)
        self.on_init()

    def update(self):
        self.texture.use()
        self.program['camPos'].write(self.camera.position)
        self.program['m_view'].write(self.camera.m_view)
        self.program['m_model'].write(self.m_model)

    def on_init(self):
        # texture
        self.texture = self.app.mesh.texture.textures[self.tex_id]
        self.program['u_texture_0'] = 0
        self.texture.use()
        # mvp
        self.program['m_proj'].write(self.camera.m_proj)
        self.program['m_view'].write(self.camera.m_view)
        self.program['m_model'].write(self.m_model)
        # light
        self.program['light.position'].write(self.app.light.position)
        self.program['light.Ia'].write(self.app.light.Ia)
        self.program['light.Id'].write(self.app.light.Id)
        self.program['light.Is'].write(self.app.light.Is)


class DynamicBaseModel(BaseModel):
    def __init__(self, app, vao_name, tex_id, pos, rot, scale, velocity, ang_velocity):
        super().__init__(app, vao_name, tex_id, pos, rot, scale, velocity, ang_velocity)
        self.on_init()

    # state vector: x, y, z, p, q, r, vx, vy, vz, p_dot, q_dot, r_dot
    # index:        0, 1, 2, 3, 4, 5, 6,  7,  8,  9,     10,    11
    @staticmethod
    def drone_dynamics(t, states, state_input, forward, right, up):
        # Constants
        mass = 0.25
        drag_factor = 0.25
        Ixx = 0.005
        Iyy = 0.005
        Izz = 0.005

        # States in body frame
        rot_matrix = glm.mat3(up, right, forward)
        u = np.zeros(6)
        # u[0:3] = np.matmul(rot_matrix, state_input[0:3])
        # u[3:6] = np.matmul(rot_matrix, state_input[3:6])
        # states[6:9] = np.matmul(rot_matrix, states[6:9])
        # states[9:12] = np.matmul(rot_matrix, states[9:12])

        # Dynamics
        x, y, z, theta, phi, psi, vx, vy, vz, p, q, r = states
        az = - 9.81*glm.sin(theta) - drag_factor/mass*vz
        ax = 9.81*glm.cos(theta)*glm.sin(phi) - drag_factor/mass*vx
        ay = 9.81*glm.cos(theta)*glm.cos(phi) - state_input[3] - drag_factor/mass*vy
        q_dot = q*r*((Iyy-Izz)/Ixx) + state_input[0]/Ixx
        r_dot = p*r*((Izz-Ixx)/Iyy) + state_input[1]/Iyy
        p_dot = p*q*((Ixx-Iyy)/Izz) + state_input[2]/Izz

        return np.concatenate([states[6:12], [ax, ay, az, q_dot, r_dot, p_dot]])

    def update(self):
        self.controller.constant_r()
        self.state_input = self.controller.output
        self.update_states()
        self.m_model = self.get_model_matrix()
        self.update_drone_vectors()
        self.m_drone_view = self.get_drone_view_matrix()

        # render object with shader
        self.texture.use()
        self.program['camPos'].write(self.camera.position)
        self.program['m_view'].write(self.camera.m_view)
        self.program['m_model'].write(self.m_model)

    def update_states(self):
        self.r.set_f_params(self.state_input, self.forward, self.right, self.up)
        self.states = np.real(self.r.integrate(self.r.t + self.app.delta_time / 1000))

    def update_drone_vectors(self):
        self.forward = glm.vec3(self.m_model * glm.vec4(0, 0, 1, 0))
        self.right = glm.vec3(self.m_model * glm.vec4(1, 0, 0, 0))
        self.up = glm.vec3(self.m_model * glm.vec4(0, 1, 0, 0))

    def get_drone_view_matrix(self):
        return glm.lookAt(glm.vec3(list(self.states[0:3])), glm.vec3(list(self.states[0:3])) + self.forward, self.up)

    def get_drone_projection_matrix(self):
        aspect_ratio = self.app.WIN_SIZE[0] / self.app.WIN_SIZE[1]
        return glm.perspective(glm.radians(50), aspect_ratio, 0.1, 100)

    def on_init(self):
        # set up the ode:
        self.r = ode(DynamicBaseModel.drone_dynamics).set_integrator('zvode', method='bdf')
        self.r.set_initial_value(self.states, self.app.time)
        # drone view matrix
        self.m_drone_view = self.get_drone_view_matrix()
        # projection matrix
        self.m_drone_proj = self.get_drone_projection_matrix()
        # texture
        self.texture = self.app.mesh.texture.textures[self.tex_id]
        self.program['u_texture_0'] = 0
        self.texture.use()
        # mvp
        self.program['m_proj'].write(self.camera.m_proj)
        self.program['m_view'].write(self.camera.m_view)
        self.program['m_model'].write(self.m_model)
        # light
        self.program['light.position'].write(self.app.light.position)
        self.program['light.Ia'].write(self.app.light.Ia)
        self.program['light.Id'].write(self.app.light.Id)
        self.program['light.Is'].write(self.app.light.Is)


class Cube(ExtendedBaseModel):
    def __init__(self, app, vao_name='cube', tex_id=0, pos=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1), velocity=(0, 0, 0), ang_velocity=(0, 0, 0)):
        super().__init__(app, vao_name, tex_id, pos, rot, scale, velocity, ang_velocity)


class Cat(ExtendedBaseModel):
    def __init__(self, app, vao_name='cat', tex_id='cat',
                 pos=(0, 0, 0), rot=(-90, 0, 0), scale=(1, 1, 1), velocity=(0, 0, 0), ang_velocity=(0, 0, 0)):
        super().__init__(app, vao_name, tex_id, pos, rot, scale, velocity, ang_velocity)


class Drone(DynamicBaseModel):
    def __init__(self, app, vao_name='drone', tex_id='drone',
                 pos=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1), velocity=(0, 0, 0), ang_velocity=(0, 0, 0)):
        super().__init__(app, vao_name, tex_id, pos, rot, scale, velocity, ang_velocity)


class SkyBox(BaseModel):
    def __init__(self, app, vao_name='skybox', tex_id='skybox',
                 pos=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1), velocity=(0, 0, 0), ang_velocity=(0, 0, 0)):
        super().__init__(app, vao_name, tex_id, pos, rot, scale, velocity, ang_velocity)
        self.on_init()

    def update(self):
        self.program['m_view'].write(glm.mat4(glm.mat3(self.camera.m_view)))

    def on_init(self):
        # texture
        self.texture = self.app.mesh.texture.textures[self.tex_id]
        self.program['u_texture_skybox'] = 0
        self.texture.use(location=0)
        # mvp
        self.program['m_proj'].write(self.camera.m_proj)
        self.program['m_view'].write(glm.mat4(glm.mat3(self.camera.m_view)))


class AdvancedSkyBox(BaseModel):
    def __init__(self, app, vao_name='advanced_skybox', tex_id='skybox',
                 pos=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1), velocity=(0, 0, 0), ang_velocity=(0, 0, 0)):
        super().__init__(app, vao_name, tex_id, pos, rot, scale, velocity, ang_velocity)
        self.on_init()

    def update(self):
        m_view = glm.mat4(glm.mat3(self.camera.m_view))
        self.program['m_invProjView'].write(glm.inverse(self.camera.m_proj * m_view))

    def on_init(self):
        # texture
        self.texture = self.app.mesh.texture.textures[self.tex_id]
        self.program['u_texture_skybox'] = 0
        self.texture.use(location=0)



















