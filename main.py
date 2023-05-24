import os
import pygame as pg
import moderngl as mgl
import sys
from model import *
from camera import Camera
from light import Light
from mesh import Mesh
from scene import Scene


class GraphicsEngine:
    def __init__(self, win_size=(1600, 900)):
        # init pygame modules
        pg.init()
        # window size
        self.WIN_SIZE = win_size
        # set opengl attr
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
        # create opengl context
        pg.display.set_mode(self.WIN_SIZE, flags=pg.OPENGL | pg.DOUBLEBUF)
        # mouse settings
        pg.event.set_grab(True)
        pg.mouse.set_visible(False)
        # detect and use existing opengl context
        self.ctx = mgl.create_context()
        # self.ctx.front_face = 'cw'
        self.ctx.enable(flags=mgl.DEPTH_TEST | mgl.CULL_FACE)
        # create an object to help track time
        self.clock = pg.time.Clock()
        self.time = 0
        self.delta_time = 0.001
        # light
        self.light = Light()
        # camera
        self.camera = Camera(self)
        # mesh
        self.mesh = Mesh(self)
        # scene
        self.scene = Scene(self)
        # drone
        self.drone = self.scene.objects[1]

    def check_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                self.save_data()
                self.mesh.destroy()
                pg.quit()
                sys.exit()

    def save_data(self):
        positions = np.array(self.drone.controller.pos_over_time)
        velocities = np.array(self.drone.controller.velocities_over_time)
        divergences = np.array(self.drone.controller.divergence_over_time)
        time = np.array(self.drone.controller.time_over_time) - self.drone.controller.time_over_time[0]
        set_points = np.array(self.drone.controller.theta_sp_over_time)
        gains = np.array(self.drone.controller.gain_over_time)
        theta = np.array(self.drone.controller.theta_over_time)
        state_inputs = np.array(self.drone.controller.state_input_over_time)
        data = np.vstack([time, positions, velocities, divergences, set_points, gains, theta, state_inputs])
        filename = os.path.join('sim_data', self.drone.controller.name)
        np.save(filename, data)

        # with open(filename, 'w') as filehandle:
        #     for i in range(len(positions)):
        #         filehandle.writelines(f"{time[i]}, {positions[i]}, {velocities[i]}, {c_variables[i]}\n")
        #     filehandle.close()

    def render(self):
        # clear framebuffer
        self.ctx.clear(color=(0.08, 0.16, 0.18))
        # render scene
        self.scene.render()
        # swap buffers
        pg.display.flip()

    def get_time(self):
        self.time = pg.time.get_ticks() * 0.001

    def run(self):
        while True:
            self.get_time()
            self.check_events()
            self.camera.update()
            self.render()
            self.delta_time = self.clock.tick(60)


if __name__ == '__main__':
    app = GraphicsEngine()
    app.run()






























