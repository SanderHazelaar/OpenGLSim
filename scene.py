from model import *
import glm


class Scene:
    def __init__(self, app):
        self.app = app
        self.objects = []
        self.load()
        # skybox
        self.skybox = AdvancedSkyBox(app)

    def add_object(self, obj):
        self.objects.append(obj)

    def load(self):
        app = self.app
        add = self.add_object

        add(Cube(app, pos=(0, 0, 0)))

        add(Drone(app, pos=(0, 0, -10), rot=(0, 0, 0), velocity=(0, 0, 0)))

    def render(self):
        for obj in self.objects:
            obj.render()
        self.skybox.render()

