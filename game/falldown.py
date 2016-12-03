import pymunk
import pygame
from pygame.locals import *
import pymunk.pygame_util
from pygame.colordict import THECOLORS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

height, width = 512, 384

# pygame initialisations
pygame.init()
screen = pygame.display.set_mode((width, height))
screen.set_alpha(None)  # alpha unused, marginal speed improvement
clock = pygame.time.Clock()

# pymunk / physics initialisations
space = pymunk.Space()
space.gravity = (0.0, -900)
draw_options = pymunk.pygame_util.DrawOptions(screen)

np.set_printoptions(threshold=np.nan, linewidth=200)


class Environment:

    def __init__(self, platform_speed=60, platform_spacing=80, gap_size=40):
        self.init_agent()
        self.init_boundaries()
        self.running = True
        self.hold_agent_stationary = True
        self.score = 0
        self.platform_spacing = platform_spacing
        self.platform_speed = platform_speed
        self.gap_size = gap_size
        self.platforms = []
        self.create_new_platform()
        self.ticks = 0

    def tick(self, action_to_take):

        # don't allow the agent to move until the first platform has passed the y-midpoint
        if self.hold_agent_stationary:
            self.check_game_begin_condition()

        # create new platform when the lowest platform height > platform_spacing
        if self.platforms[-1][0].position[1] > self.platform_spacing:
            self.create_new_platform()

        self.remove_off_screen_platforms()

        self.take_action(action_to_take)

        # update physics and draw to screen
        screen.fill(THECOLORS['white'])
        space.debug_draw(draw_options)
        space.step(1.0/60.0)
        pygame.display.update()
        clock.tick(50)
        pygame.display.set_caption("Falldown (fps: " + str(clock.get_fps()) + ", score: " + str(self.score) + ")")

        reward = self.get_reward()
        if reward != 0:
            self.score += reward


        new_state = self.format_surface_render(screen)

        self.ticks += 1
        if self.ticks > 500:
            plt.imsave('img.png', new_state, cmap=cm.gray)
            self.running = False

        # returns (new_state, reward, terminal_status)
        return new_state, reward, self.running

    def format_surface_render(self, surface):
        resized_surface = pygame.transform.smoothscale(pygame.PixelArray(surface).make_surface(), (48, 64))  # (W, H)
        formatted_array = np.zeros((64, 48))
        for i in range(64):
            for j in range(48):
                red, green, blue, alpha = resized_surface.get_at((j, i))  # (x, y)
                formatted_array[i, j] = (red + green + blue) / 3
        return formatted_array

    def take_action(self, a):
        # action == [0, 0] : do not change agent velocity
        # action == [0, 1] : left
        # action == [1, 0] : right
        if a == [0, 1]:
            self.agent_body.apply_impulse_at_local_point((-15.0, 0.0), (0.0, 0.0))
        elif a == [1, 0]:
            self.agent_body.apply_impulse_at_local_point((15.0, 0.0), (0.0, 0.0))

    def get_reward(self):
        # punishment if agent hits the roof (terminal case).
        if self.agent_body.position[1] > height-15:
            self.running = False
            return -self.score*0.9
        # reward if agent passes through a gap it hasn't passed through before.
        else:
            for platform in self.platforms:
                if (not platform[-1]) and (self.agent_body.position[1] < platform[0].position[1]):
                    platform[-1] = True
                    return 10
        # else, reward 0.
        return 0

    def create_new_platform(self):
        # add new platform with between 60% change gap in middle, 40% chance at either end.
        if np.random.random() < 0.4:
            if np.random.random() > 0.5:
                platform_start = 4
                platform_end = width - self.gap_size
            else:
                platform_start = self.gap_size
                platform_end = width - 4
            platform_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
            platform_body.position = (0.0, 0.0)
            platform_shape = pymunk.Poly(platform_body, [(platform_start,0.0),
                                                         (platform_start,5.0),
                                                         (platform_end, 0.0),
                                                         (platform_end, 5.0)])
            platform_shape.color = THECOLORS["black"]
            platform_shape.elasticity = 1
            platform_body._set_velocity_func(self.platform_velocity_function())
            space.add(platform_body, platform_shape)
            self.platforms.append([platform_body, platform_shape, False])
        else:
            line_1_start = 4
            line_1_end = np.random.randint(self.gap_size+4, width-(self.gap_size*2)-4)
            line_2_start = line_1_end + self.gap_size
            line_2_end = width-4
            platform_1_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
            platform_1_body.position = (0.0, 0.0)
            platform_1_shape = pymunk.Poly(platform_1_body, [(line_1_start,0.0),
                                                         (line_1_start,5.0),
                                                         (line_1_end, 0.0),
                                                         (line_1_end, 5.0)])
            platform_1_shape.color = THECOLORS["black"]
            platform_1_shape.elasticity = 1
            platform_1_body._set_velocity_func(self.platform_velocity_function())
            platform_2_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
            platform_2_body.position = (0.0, 0.0)
            platform_2_shape = pymunk.Poly(platform_2_body, [(line_2_start,0.0),
                                                         (line_2_start,5.0),
                                                         (line_2_end, 0.0),
                                                         (line_2_end, 5.0)])
            platform_2_shape.color = THECOLORS["black"]
            platform_2_shape.elasticity = 1
            platform_2_body._set_velocity_func(self.platform_velocity_function())
            space.add(platform_1_body, platform_1_shape, platform_2_body, platform_2_shape)
            self.platforms.append([platform_1_body, platform_1_shape, platform_2_body,
                                   platform_2_shape, False])

    def remove_off_screen_platforms(self):
        # if bottom of platform y-coordinate is less than 0, remove it from the list and space.
        if self.platforms[0][0].position[1] > height:
            if len(self.platforms[0]) > 3:
                space.remove(self.platforms[0][0], self.platforms[0][1],
                             self.platforms[0][2], self.platforms[0][3])
            else:
                space.remove(self.platforms[0][0], self.platforms[0][1])
            self.platforms.pop(0)

    def check_game_begin_condition(self):
        # self.agent_body.position = (width/2, height/2)
        if self.platforms[0][0].position[1] > 80:
            self.agent_body._set_velocity_func(self.agent_velocity_function())
            self.hold_agent_stationary = False

    # velocity integration function for the platforms.
    def platform_velocity_function(self):
        def f(body, gravity, damping, dt):
            body._set_velocity((0.0, self.platform_speed))
        return f

    # agent velocity integration function to allow for x dampening (friction producing weird behaviour)
    def agent_velocity_function(self):
        def f(body, gravity, damping, dt):
            body._set_velocity((body.velocity.x * 0.975, body.velocity.y+(gravity[1]*dt)))
        return f

    # to hold the agent stationary initially (v. bad to _set_velocity at each time step)
    def agent_stationary_velocity_function(self):
        def f(body, gravity, damping, dt):
            body._set_velocity((0.0, 0.0))
        return f

    def init_agent(self):
        self.agent_body = pymunk.Body(mass=1, moment=1)
        self.agent_body.position = (width/2, height/2)
        self.agent_shape = pymunk.Circle(self.agent_body, 15)
        self.agent_shape.color = THECOLORS["red"]
        self.agent_shape.elasticity = 0.2
        self.agent_body._set_velocity_func(self.agent_stationary_velocity_function())
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
            wall.color = THECOLORS['white']
        space.add(walls)

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
