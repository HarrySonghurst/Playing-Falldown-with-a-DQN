import pymunk
import pygame
from pygame.locals import *
import pymunk.pygame_util
from pygame.colordict import THECOLORS
import numpy as np



height, width = 512, 384
obstacle_spacing = 50

# pygame initialisations
pygame.init()
screen = pygame.display.set_mode((width, height))
screen.set_alpha(None)  # alpha unused, marginal speed improvement
clock = pygame.time.Clock()


# pymunk / physics initialisations
space = pymunk.Space()
space.gravity = (0.0, -900)
draw_options = pymunk.pygame_util.DrawOptions(screen)


class Environment:

    def __init__(self):
        self.init_agent()
        self.init_boundaries()
        self.running = True
        self.hold_stationary = True
        # init list of obstacles and create the first one
        self.obstacles = []
        self.create_new_obstacle()
        pass

    def tick(self, action_to_take):

        # initially, don't allow the agent to move, and hold the body still until the first obstacle
        # has passed the vertical midpoint of the screen.
        # take_action()
        # if the last obstacle in the list of obstacles has passed an appropriate height, create_new_obstacle()
        # remove_off_screen_obstacles()

        # don't allow the agent to move until the first obstacle has passed the y-midpoint
        # if not self.hold_stationary:
        #     self.check_game_begin_condition()
        # if self.hold_stationary:
        #     self.agent_body.position = (width/2, height/2)
        #     self.agent_body.velocity = (0.0, 0.0)

        if self.obstacles[-1][0].position[1] > 50:
            self.create_new_obstacle()

        self.take_action(action_to_take)
        screen.fill(THECOLORS["white"])
        space.debug_draw(draw_options)
        space.step(1.0/60.0)
        pygame.display.flip()
        clock.tick(50)
        pygame.display.set_caption("Falldown (fps: " + str(clock.get_fps()) + ")")

        # get_reward returns the reward associated with the new state, along with terminal_status
        # new_state = format_surface_render(pygame.PixelArray(surface))

        # returns (new_state, reward, terminal_status)
        pass

    def format_surface_render(self, pixel_array):
        # transform and return the pixel array obtained from the pygame surface by scaling it down, then
        # turning it into a 1D array to be fed into the neural network.
        pass

    def take_action(self, a):
        # action == [0, 0] : do not change agent velocity
        # action == [0, 1] : decrease agent x velocity
        # action == [1, 0] : increase agent x velocity
        if a == [0, 1]:
            self.agent_body.apply_impulse_at_local_point((-10.0, 0.0), (0.0, 0.0))
        elif a == [1, 0]:
            self.agent_body.apply_impulse_at_local_point((10.0, 0.0), (0.0, 0.0))
        pass

    def get_reward(self):
        # check whether the agents y-position is less than or equal to its radius, if true,
        # return -100 (maybe -(total score)?).
        # check whether the y-coordinate of the top of the agent is greater than all the bottom
        # y-coordinates of the obstacles in the list, also check if the 'passed' boolean on
        # the obstacle is false, if it is, then return 10 and set the boolean true.
        # else return -0.1 to encourage speed
        # returns tuple of (double reward, boolean terminal_status)
        pass

    def create_new_obstacle(self):
        # add new pymunk body to the space that overrides gravity, give it a constant velocity upwards, friction > 0.
        # append the body to a list of obstacle bodies in a tuple with a 'passed' boolean set to false.
        # A kinematic body is an hybrid body which is not affected by forces and collisions like
        # a static body but can moved with a linear velocity like a dynamic body.
        obstacle_body = pymunk.Body(mass=1, moment=1, body_type=pymunk.Body.KINEMATIC)
        obstacle_body.position = (0.0, 0.0)
        obstacle_shape = pymunk.Poly(obstacle_body, [(5.0,5.0),(5.0,10.0),(width-5, 5.0), (width-5, 10.0)])
        # obstacle_shape = pymunk.Segment(obstacle_body, (5.0, ))
        obstacle_shape.color = THECOLORS["black"]
        obstacle_shape.friction = 0.5
        obstacle_shape.elasticity = 0.1
        obstacle_body._set_velocity_func(self.obstacle_velocity_function())
        space.add(obstacle_body, obstacle_shape)
        self.obstacles.append((obstacle_body, False))
        pass

    # velocity integration function for the obstacles.
    def obstacle_velocity_function(self):
        def f(body, gravity, damping, dt):
            body._set_velocity((0.0, 20))
        return f

    def remove_off_screen_obstacles(self):
        # if bottom of obstacle y-coordinate is less than 0, remove it from the list and space.
        pass

    def init_agent(self):
        self.agent_body = pymunk.Body(mass=1, moment=1)
        self.agent_body.position = (width/2, height/2)
        self.agent_shape = pymunk.Circle(self.agent_body, 15)
        self.agent_shape.color = THECOLORS["red"]
        self.agent_shape.elasticity = 0.1
        self.agent_shape.friction = 0.0
        space.add(self.agent_body, self.agent_shape)

    def init_boundaries(self):
        static_body = space.static_body
        walls = [
                 pymunk.Segment(static_body, (0.0, 0.0), (0.0, height), 0.0),          # left wall
                 pymunk.Segment(static_body, (width-1, 0.0), (width-1, height), 0.0),  # right wall
                 pymunk.Segment(static_body, (0.0, 1.0), (width, 1.0), 0.0),           # floor
                 pymunk.Segment(static_body, (0.0, height), (width, height), 0.0)      # ceiling
                ]
        for wall in walls:
            wall.elasticity = 0.1
            wall.friction = 0.50
        space.add(walls)

    def check_game_begin_condition(self):
        pass

if __name__ == "__main__":
    environment = Environment()

    # for the purposes of testing the game, keys pressed must be held in a dictionary and passed
    # to tick otherwise holding a key has no effect

    left, right = False, False

    while environment.running:

        # key handling for testing
        for event in pygame.event.get():
            if event.type == QUIT:
                environment.running = False
            if event.type == KEYDOWN:
                if event.key == K_LEFT:
                    left = True
                elif event.key == K_RIGHT:
                    right = True
            elif event.type == KEYUP:
                if event.key == K_LEFT:
                    left = False
                elif event.key == K_RIGHT:
                    right = False
        if left and right:
            left, right = False, False

        if left:
            action = [0, 1]
        elif right:
            action = [1, 0]
        else:
            action = [0, 0]

        environment.tick(action)
