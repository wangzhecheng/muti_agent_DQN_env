""""""
"""
This script build the 'HunterPrey' class. To initialize it, two arguments:
num_hunters and num_preys are required. num_hunters is the number of the hunters
in the game, and num_preys is the number of the preys in the game. Everytime a
hunter KILL a prey, that hunter will receive some reward, and other hunters will
receive some penalties (negative rewards). The game terminates when all preys are
killed, and then a new game is initialized.

You can externally control the hunters by input action for each hunter, but you can
not control preys externally. They are internally programmed to move for escaping
from the hunters."""

import hunter_prey_utils_2
import pygame
import numpy as np
import sys
import random
import os
import skimage.io
import skimage.transform
import skimage.color
import matplotlib.pyplot as plt

import pygame.surfarray as surfarray
from pygame.locals import *
from itertools import cycle

FPS = 20 # frame per second
SCREENWIDTH = 180
SCREENHEIGHT = 180

# configs
REWARD = 1.0 # reward for the hunter killing a prey
BONUS = 10.0
BAD_REWARD = -0.1 # Hunters receive  reward for each move when they do not capture a prey
PENALTY = -10.0 # penalty for a hunter if other hunter kill a prey
HUNTER_SPEED = 4 # moving speed of hunters (pixels/frame)
PREY_SPEED = 10 # moving speed of preys (pixels/frame)
RAND_MOVE_PROB = 0.01 # the probability of a PREY to move randomly
DETECTION_RADIUS = 90 # the detection radius for hunter
# initial the game

#os.environ["SDL_VIDEODRIVER"] = "dummy"
pygame.init()
FPSCLOCK = pygame.time.Clock()
SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
pygame.display.set_caption('Hunter Prey')

IMAGES = hunter_prey_utils_2.load()

HUNTER_WIDTH = IMAGES['hunter'].get_width()
HUNTER_HEIGHT = IMAGES['hunter'].get_height()
PREY_WIDTH = IMAGES['prey_self'].get_width()
PREY_HEIGHT = IMAGES['prey_self'].get_height()
BONUS_SIZE = IMAGES['bonus'].get_height()

class HunterPrey:
    def __init__(self, num_hunters, num_preys):
        """
        self.hunter_list: a list containing all hunters, each hunter is a dict,
        where hunter['x'] is the x coordinate, and hunter['y'] is the y coordinate
        of that hunter. Coordinates are randomly initialized.

        self.prey_list: a list containing all preys. Similiar to hunter, each prey
        is a dict containing x and y coordinates.Coordinates are randomly initialized.

        self.terminal: a boolean, True if game over, and False otherwise.

        Arguments:
            num_hunters: int, number of hunters in the game.
            num_preys: int, number of preys in the game.
        """
        self.FPS = FPS
        self.num_hunters = num_hunters
        self.num_preys = num_preys
        self.initial_num_prey = num_preys
        # Construct hunter list
        self.hunter_list = []
        for i in xrange(num_hunters):
            hunter = {}
            hunter['x'] = random.randint(0, int(SCREENWIDTH - HUNTER_WIDTH))
            hunter['y'] = random.randint(0, int(SCREENHEIGHT - HUNTER_HEIGHT))
            self.hunter_list.append(hunter)

        # Construct prey list
        self.prey_list = []
        self.isalive = []
        for i in xrange(num_preys):
            prey = {}
            prey['x'] = random.randint(0, int(SCREENWIDTH - PREY_WIDTH))
            prey['y'] = random.randint(0, int(SCREENHEIGHT - PREY_HEIGHT))
            self.prey_list.append(prey)
            self.isalive.append(True)

        # total score
        self.total_score = 0

        self.nearest_prey_list = []
        self.continous_moving_step = []

        # bonus list
        self.bonus_list = []

        for i in xrange(num_hunters):
            self.nearest_prey_list.append(random.randint(0, num_preys-1))
            self.continous_moving_step.append(random.randint(7, 12))

    def step(self, action_list):
        """
        This function serves as a game simulator. Receive actions, return
         rewards and next states.

        Argument:
            action_list: a list with length=num_hunters, each action in
            action_list is an int from 0 to 4.
            0: left, 1: up, 2: right, 3: down, 4: don't move
        Return:
            states: a list with length=num_hunters. For each hunter, the state
            is an 200*200*3 image where the hunter itself is black and other
            hunters are grey.
            reward: a list with length=num_hunters. Each entry corresponds to
            the one-frame reward for each hunter.
            self.terminal: boolean. True if game over.
        """

        # check action_list
        if not len(action_list) == self.num_preys:
            raise ValueError('number of actions != number of hunters')
        if max(action_list) > 4 or min(action_list) < 0:
            raise ValueError('action index must be int from 0 to 4!!')

        # append bonus
        if random.random() < 1.0 and len(self.bonus_list) <= 1:
            bonus = {}
            bonus['x'] = random.randint(0, int(SCREENWIDTH - BONUS_SIZE))
            bonus['y'] = random.randint(0, int(SCREENHEIGHT - BONUS_SIZE))
            self.bonus_list.append(bonus)

        # preys' movement
        for i in xrange(self.num_preys):
            if not self.isalive[i]:
                continue
            action = action_list[i]
            if action == 0:
                # move left
                self.prey_list[i]['x'] -= PREY_SPEED
                # Don't let it outbound
                if self.prey_list[i]['x'] <= 0:
                    self.prey_list[i]['x'] = 0
            elif action == 1:
                # move up
                self.prey_list[i]['y'] -= PREY_SPEED
                # Don't let it outbound
                if self.prey_list[i]['y'] <= 0:
                    self.prey_list[i]['y'] = 0
            elif action == 2:
                # move right
                self.prey_list[i]['x'] += PREY_SPEED
                # Don't let it outbound
                if self.prey_list[i]['x'] >= int(SCREENWIDTH - PREY_WIDTH):
                    self.prey_list[i]['x'] = int(SCREENWIDTH - PREY_WIDTH)
            elif action == 3:
                # move down
                self.prey_list[i]['y'] += PREY_SPEED
                # Don't let it outbound
                if self.prey_list[i]['y'] >= int(SCREENHEIGHT - PREY_HEIGHT):
                    self.prey_list[i]['y'] = int(SCREENHEIGHT - PREY_HEIGHT)

        # HUNTERs' movement (Move towards the direction of the NEAREST hunter)
        for i in xrange(self.num_hunters):
            # with prob RAND_MOVE_PROB, a prey moves randomly:
            if random.random() <  RAND_MOVE_PROB:
                hunter_action = random.randint(0, 4)
            else:
                # Find the nearest prey
                hunter = self.hunter_list[i]

                if self.continous_moving_step[i] > 0 and self.isalive[self.nearest_prey_list[i]] == True and random.random() < 0.8:
                    self.continous_moving_step[i] -= 1
                    nearest_prey = self.prey_list[self.nearest_prey_list[i]]
                else:
                    #for i in xrange(self.num_preys):
                    #    if self.isalive[i]:
                    #        alive_preys.append(self.prey_list[i])
                    # list(enumerate(self.prey_list))
                    sorted_enumerate = sorted(list(enumerate(self.prey_list)), key=lambda w: abs(w[1]['x'] - hunter['x']) + abs(w[1]['y'] - hunter['y']))
                    while True:
                        nearest_prey_idx = sorted_enumerate[0][0]
                        nearest_prey = sorted_enumerate[0][1]
                        if self.isalive[nearest_prey_idx]:
                            break
                        sorted_enumerate = sorted_enumerate[1:]
                    self.nearest_prey_list[i] = nearest_prey_idx
                    self.continous_moving_step[i] = random.randint(7, 12)

                # Find the best moving direction
                x_distance = abs(nearest_prey['x'] - hunter['x'])
                y_distance = abs(nearest_prey['y'] - hunter['y'])

                # if the nearest hunter is out of detection range, move randomly
                if x_distance ** 2 + y_distance ** 2 > DETECTION_RADIUS ** 2 and self.continous_moving_step[i] >= 0:
                    hunter_action = random.randint(0, 4)

                #if np.sqrt((nearest_prey['x']-hunter['x'])**2 + (nearest_prey['y']-hunter['y'])**2) > 50:
                #    continue
                else:
                    if nearest_prey['x'] < hunter['x']:
                        # prey need to move left
                        move_left = True
                    else:
                        move_left = False

                    if nearest_prey['y'] < hunter['y']:
                        # prey need to move up
                        move_up = True
                    else:
                        move_up = False

                    if move_up == True and move_left == True:
                        if x_distance > y_distance:
                            hunter_action = 0
                        else:
                            hunter_action = 1
                    elif move_up == True and move_left == False:
                        if x_distance > y_distance:
                            hunter_action = 2
                        else:
                            hunter_action = 1
                    elif move_up == False and move_left == True:
                        if x_distance > y_distance:
                            hunter_action = 0
                        else:
                            hunter_action = 3
                    else:
                        if x_distance > y_distance:
                            hunter_action = 2
                        else:
                            hunter_action = 3

            # After the moving direction is determined, move each prey just the same way as moving hunters.
            if hunter_action == 0:
                # move left
                self.hunter_list[i]['x'] -= HUNTER_SPEED
                if self.hunter_list[i]['x'] <= 0:
                    self.hunter_list[i]['x'] = 0
            elif hunter_action == 1:
                # move up
                self.hunter_list[i]['y'] -= HUNTER_SPEED
                if self.hunter_list[i]['y'] <= 0:
                    self.hunter_list[i]['y'] = 0
            elif hunter_action == 2:
                # move right
                self.hunter_list[i]['x'] += HUNTER_SPEED
                if self.hunter_list[i]['x'] >= int(SCREENWIDTH - HUNTER_WIDTH):
                    self.hunter_list[i]['x'] = int(SCREENWIDTH - HUNTER_WIDTH)
            elif hunter_action == 3:
                # move down
                self.hunter_list[i]['y'] += HUNTER_SPEED
                if self.hunter_list[i]['y'] >= int(SCREENHEIGHT - HUNTER_HEIGHT):
                    self.hunter_list[i]['y'] = int(SCREENHEIGHT - HUNTER_HEIGHT)

        # Check if any hunter kill any prey, and then assign rewards.
        reward_list = [] # len = num_preys
        terminal_list = []
        for i in xrange(self.num_preys):
            reward_list.append(0.0)
            terminal_list.append(False)
        #killed_set = set() # A set containing the indices of killed preys
        for i in xrange(self.num_preys):
            if not self.isalive[i]:
                continue
            is_killed = False
            for j in xrange(self.num_hunters):
                if self.is_killed(self.hunter_list[j], self.prey_list[i]): # is_killed will return True if that hunter kill that prey.
                    #killed_set.add(i)
                    # reward that hunter
                    reward_list[i] = PENALTY
                    is_killed = True
                    self.isalive[i] = False
                    terminal_list[i] = True

            if is_killed == False:
                reward_list[i] = REWARD

        for i in xrange(self.num_preys):
            if not self.isalive[i]:
                continue
            prey = self.prey_list[i]
            for bonus in self.bonus_list:
                if np.sqrt((bonus['x']-prey['x'])**2 + (bonus['y']-prey['y'])**2) <= (BONUS_SIZE + PREY_HEIGHT)/2:
                    self.bonus_list.remove(bonus)
                    reward_list[i] += BONUS
                    self.total_score += BONUS


        # remove killed preys from prey list.
        #self.prey_list = [m for n, m in enumerate(self.prey_list) if n not in killed_set]
        #self.num_preys = len(self.prey_list)

        total_score = self.total_score
        isalive = self.isalive
        # Check if terminal. If there is no prey in the prey_list, then terminal=True, and initialize the game.
        if sum(self.isalive) == 0:
            initial_num_hunter = self.num_hunters
            initial_num_prey = self.initial_num_prey
            over = True
            former_FPS = self.FPS
            self.__init__(num_hunters=initial_num_hunter, num_preys=initial_num_prey)
            self.FPS = former_FPS
        else:
            over = False
            self.total_score += 1

        # Draw images and build state for each player (hunter)
        SCREEN.blit(IMAGES['background'], (0, 0))
        for hunter in self.hunter_list:
            SCREEN.blit(IMAGES['hunter'], (hunter['x'], hunter['y']))
        for bonus in self.bonus_list:
            SCREEN.blit(IMAGES['bonus'], (bonus['x'], bonus['y']))

        states = []
        for i in xrange(self.num_preys):
            for j in xrange(self.num_preys):
                # The prey itself is black
                if self.isalive[j] == False:
                    continue
                if j == i:
                    SCREEN.blit(IMAGES['prey_self'], (self.prey_list[j]['x'], self.prey_list[j]['y']))
                # Other hunters is grey
                else:
                    SCREEN.blit(IMAGES['prey_other'], (self.prey_list[j]['x'], self.prey_list[j]['y']))
            # The state for each hunter
            image_data = pygame.surfarray.array3d(pygame.display.get_surface())
            states.append(image_data)

        for j in xrange(self.num_preys):
            if self.isalive[j] == False:
                continue
            SCREEN.blit(IMAGES['prey_self'], (self.prey_list[j]['x'], self.prey_list[j]['y']))

        """# print score on screen
        message = 'AI score: %d' % self.score
        font = pygame.font.Font(None, 40)
        text = font.render(message, 1, (50, 50, 50))
        SCREEN.blit(text, (10, 10))
        message2 = 'Highest: %d' % self.highest_score
        font = pygame.font.Font(None, 40)
        text2 = font.render(message2, 1, (50, 50, 50))
        SCREEN.blit(text2, (10, 40))"""

        # Update the view
        pygame.display.update()
        # Update time
        FPSCLOCK.tick(self.FPS)

        return states, reward_list, terminal_list, isalive, total_score, over


    def is_killed(self, hunter, prey):
        """
        Arguments:
            hunter: a dict, containing hunter['x'] and hunter['y']
            prey: a dict, containing prey['x'] and prey['y']
        Return:
            Boolean. True if killed and False otherwise.
            """
        dist = np.sqrt((hunter['x']-prey['x'])**2 + (hunter['y']-prey['y'])**2)
        # if the L2 distance between hunter and prey is less than the sum of their radius, then killed.
        if dist < (HUNTER_HEIGHT + PREY_HEIGHT)/2:
            return True
        else:
            return False

def human_play():
    num_hunters =2
    num_preys =2
    game_state = HunterPrey(num_hunters=num_hunters, num_preys=num_preys)
    t = 0
    while 1:
        action = []
        for i in xrange(num_preys):
            action.append(random.randint(0, 4))
        # receive the keyboard input for the action for hunter 0.
        if pygame.key.get_pressed()[pygame.K_LEFT]:
            action[1] = 0
        elif pygame.key.get_pressed()[pygame.K_UP]:
            action[1] = 1
        elif pygame.key.get_pressed()[pygame.K_RIGHT]:
            action[1] = 2
        elif pygame.key.get_pressed()[pygame.K_DOWN]:
            action[1] = 3
        else:
            action[1] = 4
        pygame.event.poll()
        states, reward, terminal, isalive_list, total_score, over = game_state.step(action)
        if over:
            print total_score
        plt.subplot(1, 2, 1)
        plt.imsave('prey1.jpg', states[0])
        plt.imshow(skimage.color.rgb2grey(skimage.transform.resize(states[0], (80, 80))), cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imsave('prey2.jpg', states[1])
        plt.imshow(skimage.color.rgb2grey(skimage.transform.resize(states[1], (80, 80))), cmap='gray')
        plt.show()
        """print ('##############')
        print ('reward list: ' + str(reward))
        print ('terminal list: ' + str(terminal))
        print ('alive list: '+str(isalive_list))
        print ('total score: ' + str(total_score))
        print ('over: '+str(over))
        plt.subplot(1, 2, 1)
        plt.imshow(skimage.color.rgb2grey(skimage.transform.resize(states[0], (36, 36))), cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(skimage.color.rgb2grey(skimage.transform.resize(states[1], (36, 36))), cmap='gray')
        plt.show()"""

        t += 1

if __name__ == "__main__":
    human_play()