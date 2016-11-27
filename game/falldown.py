class Enviroment:

    def __init__(self):
        # pymunk space with gravity = 9.8
        # init list of obstacles and create the first one
        # init the agent (x-velocity, position).
        # tick()
        pass

    def tick(self, action):

        # initially, don't allow the agent to move, and hold the body still until the first obstacle
        # has passed the vertical midpoint of the screen.
        # take_action()
        # if the last obstacle in the list of obstacles has passed an appropriate height, create_new_obstacle()

        # pymunk draw(surface, space)
        # pymunk space.step
        # pygame clock tick

        # get_reward returns the reward associated with the new state, along with terminal_status
        # new_state = format_surface_render(pygame.PixelArray(surface))

        # returns (new_state, reward, terminal_status)
        pass

    def format_surface_render(self, pixel_array):
        # transform and return the pixel array obtained from the pygame surface by scaling it down, then
        # turning it into a 1D array to be fed into the neural network.
        pass

    def take_action(self, action):
        # action == [0, 0] : do not change agent velocity
        # action == [0, 1] : increase agent x velocity
        # action == [1, 0] : decrease agent x velocity
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
        pass

    def remove_off_screen_obstacles(self):
        # if bottom of obstacle y-coordinate is less than 0, remove it from the list and space.
        pass

