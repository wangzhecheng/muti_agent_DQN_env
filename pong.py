import pong_utils
import pygame
import skimage.transform
import matplotlib.pyplot as plt
import numpy as np
import sys
import random
import time
import os
import pygame.surfarray as surfarray
from pygame.locals import *
from itertools import cycle
import skimage.io
import skimage.color


FPS = 20  # frame per second
SCREENWIDTH = 240
SCREENHEIGHT = 240

# configs
REWARD = 10.0
PENALTY = -10.0
PAD_SPEED_1 = 17
PAD_SPEED_2 = 17
BALL_SPEED_X = 8
BALL_SPEED_Y_LIMIT = 10
BALL_SPEED_X_LIMIT = 10
AUTOMOVE_RANDOM_FLIP_RATE = 0.15

# initial the gameS

#os.environ["SDL_VIDEODRIVER"] = "dummy"
pygame.init()
FPSCLOCK = pygame.time.Clock()
SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
pygame.display.set_caption('Pong')

IMAGES = pong_utils.load()

PAD_WIDTH = IMAGES['paddle_self'].get_width()
PAD_HEIGHT = IMAGES['paddle_self'].get_height()
BALL_SIZE = IMAGES['ball'].get_width()

class Pong:
    def __init__(self, two_players=True):
        self.two_players = two_players
        # paddle positions
        self.pad1_X = 10                  # right side of paddle 1
        self.pad2_X = SCREENWIDTH - 10     # left side of paddle 2
        self.pad1_Y = SCREENHEIGHT / 2       # y center of paddle 1
        self.pad2_Y = SCREENWIDTH / 2       # y center of paddle 2
        # paddle speed
        self.pad1_vel = 0
        self.pad2_vel = 0
        # ball position and speed
        if random.random() < 0.5:
            self.ball_X = SCREENWIDTH * 3 / 4        # x center of ball
            self.ball_vel_X = -BALL_SPEED_X
        else:
            self.ball_X = SCREENWIDTH * 1 / 4
            self.ball_vel_X = BALL_SPEED_X
        self.ball_Y = SCREENHEIGHT / 2       # y center of ball

        self.ball_vel_Y = random.randint(-int(BALL_SPEED_X * 1/3), int(BALL_SPEED_X * 1/3))

        self.at_wall = False
        self.at_paddle = False
        self.total_score_1 = 0
        self.total_score_2 = 0

    def step(self, action_list):
        """
        action is a list of length 1 or 2. Each element is the action index.
        0: Not move, 1: move up, 2: move down
        """
        if self.two_players:
            assert len(action_list) == 2
        else:
            assert len(action_list) == 1

        if max(action_list) > 2 or min(action_list) < 0:
            raise ValueError('action index must be int from 0 to 2!!')

        # paddles' movements
        if action_list[0] == 1:
            self.pad1_vel = -PAD_SPEED_1
        elif action_list[0] == 2:
            self.pad1_vel = PAD_SPEED_1
        else:
            self.pad1_vel = 0

        self.pad1_Y += self.pad1_vel
        if self.pad1_Y >= SCREENHEIGHT - PAD_HEIGHT / 2:
            self.pad1_Y = SCREENHEIGHT - PAD_HEIGHT / 2
        if self.pad1_Y <= PAD_HEIGHT / 2:
            self.pad1_Y = PAD_HEIGHT / 2


        if self.two_players:
            if action_list[1] == 1:
                self.pad2_vel = -PAD_SPEED_2
            elif action_list[1] == 2:
                self.pad2_vel = PAD_SPEED_2
            else:
                self.pad2_vel = 0

            self.pad2_Y += self.pad2_vel
            if self.pad2_Y >= SCREENHEIGHT - PAD_HEIGHT / 2:
                self.pad2_Y = SCREENHEIGHT - PAD_HEIGHT / 2
            if self.pad2_Y <= PAD_HEIGHT / 2:
                self.pad2_Y = PAD_HEIGHT / 2

        else: #Automatic move
            if self.ball_vel_X > 0 and self.ball_vel_X > 2/3 * SCREENWIDTH:
                if self.pad2_Y > self.ball_Y:
                    self.pad2_vel = -PAD_SPEED_2
                elif self.pad2_Y < self.ball_Y:
                    self.pad2_vel = PAD_SPEED_2
                else:
                    self.pad2_vel = 0
            else:
                pass
            if random.random() < AUTOMOVE_RANDOM_FLIP_RATE:
                self.pad2_vel = -self.pad2_vel


            self.pad2_Y += self.pad2_vel
            if self.pad2_Y >= SCREENHEIGHT - PAD_HEIGHT / 2:
                self.pad2_Y = SCREENHEIGHT - PAD_HEIGHT / 2
                self.pad2_vel = -self.pad2_vel
            if self.pad2_Y <= PAD_HEIGHT / 2:
                self.pad2_Y = PAD_HEIGHT / 2
                self.pad2_vel = -self.pad2_vel

        # ball's movement
        self.ball_X += self.ball_vel_X
        self.ball_Y += self.ball_vel_Y

        # run into the wall
        if self.at_wall:
            self.at_wall = False

        if self.ball_Y <= BALL_SIZE/2 and self.at_wall == False:
            if self.ball_vel_Y != 0:
                self.ball_X -= (BALL_SIZE/2 - self.ball_Y) * self.ball_vel_X /self.ball_vel_Y
            self.ball_Y = BALL_SIZE/2
            self.at_wall = True
            # bounce
            self.ball_vel_Y = - self.ball_vel_Y

        if self.ball_Y >= SCREENHEIGHT - BALL_SIZE/2 and self.at_wall == False:
            if self.ball_vel_Y != 0:
                self.ball_X -= (self.ball_Y - SCREENHEIGHT + BALL_SIZE/2) * self.ball_vel_X / self.ball_vel_Y
            self.ball_Y = SCREENHEIGHT - BALL_SIZE/2
            self.at_wall = True
            # bounce
            self.ball_vel_Y = - self.ball_vel_Y

        # run into paddle

        if self.at_paddle:
            self.at_paddle = False

        # run into paddle 1
        if self.ball_X <= self.pad1_X + BALL_SIZE / 2:
            virtual_Y = self.ball_Y - (self.pad1_X + BALL_SIZE/2 - self.ball_X) * self.ball_vel_Y /self.ball_vel_X
            if (self.pad1_Y + PAD_HEIGHT/2 + BALL_SIZE/2 > virtual_Y > self.pad1_Y - PAD_HEIGHT/2 - BALL_SIZE/2) and self.at_paddle == False:
                self.ball_Y = virtual_Y
                self.ball_X = self.pad1_X + BALL_SIZE / 2
                self.at_paddle = True
                # speed change
                self.ball_vel_X = -self.ball_vel_X
                if abs(self.ball_vel_X) < BALL_SPEED_X_LIMIT:
                    if self.ball_vel_X > 0:
                        self.ball_vel_X += abs(self.pad1_vel / 9)
                    else:
                        self.ball_vel_X -= abs(self.pad1_vel / 9)

                self.ball_vel_Y += self.pad1_vel
                if abs(self.ball_vel_Y) >= BALL_SPEED_Y_LIMIT:
                    if self.ball_vel_Y > 0:
                        self.ball_vel_Y = BALL_SPEED_Y_LIMIT
                    else:
                        self.ball_vel_Y = -BALL_SPEED_Y_LIMIT

        # run into paddle 2
        if self.ball_X >= self.pad2_X - BALL_SIZE / 2:
            virtual_Y = self.ball_Y - (-self.pad2_X + BALL_SIZE/2 + self.ball_X) * self.ball_vel_Y /self.ball_vel_X
            if (self.pad2_Y + PAD_HEIGHT/2 + BALL_SIZE/2 > virtual_Y > self.pad2_Y - PAD_HEIGHT/2 - BALL_SIZE/2) and self.at_paddle == False:
                self.ball_Y = virtual_Y
                self.ball_X = self.pad2_X - BALL_SIZE / 2
                self.at_paddle = True
                # speed change
                self.ball_vel_X = -self.ball_vel_X
                if abs(self.ball_vel_X) < BALL_SPEED_X_LIMIT:
                    if self.ball_vel_X > 0:
                        self.ball_vel_X += abs(self.pad2_vel / 9)
                    else:
                        self.ball_vel_X -= abs(self.pad2_vel / 9)

                self.ball_vel_Y += self.pad2_vel
                if abs(self.ball_vel_Y) >= BALL_SPEED_Y_LIMIT:
                    if self.ball_vel_Y > 0:
                        self.ball_vel_Y = BALL_SPEED_Y_LIMIT
                    else:
                        self.ball_vel_Y = -BALL_SPEED_Y_LIMIT
        if self.ball_vel_Y == 0:
            self.ball_vel_Y = 1

        # Draw image and get state for each player
        SCREEN.blit(IMAGES['background'], (0, 0))
        SCREEN.blit(IMAGES['ball'], (self.ball_X - BALL_SIZE / 2, self.ball_Y - BALL_SIZE / 2))

        # state1
        SCREEN.blit(IMAGES['paddle_self'], (self.pad1_X - PAD_WIDTH / 2, self.pad1_Y - PAD_HEIGHT / 2))
        SCREEN.blit(IMAGES['paddle_other'], (self.pad2_X - PAD_WIDTH / 2, self.pad2_Y - PAD_HEIGHT / 2))
        state1 = pygame.surfarray.array3d(pygame.display.get_surface())

        # state2
        SCREEN.blit(IMAGES['paddle_other'], (self.pad1_X - PAD_WIDTH / 2, self.pad1_Y - PAD_HEIGHT / 2))
        SCREEN.blit(IMAGES['paddle_self'], (self.pad2_X - PAD_WIDTH / 2, self.pad2_Y - PAD_HEIGHT / 2))
        state2 = pygame.surfarray.array3d(pygame.display.get_surface())

        state2 = np.flip(state2, axis=0)

        SCREEN.blit(IMAGES['paddle_self'], (self.pad1_X - PAD_WIDTH / 2, self.pad1_Y - PAD_HEIGHT / 2))
        SCREEN.blit(IMAGES['paddle_self'], (self.pad2_X - PAD_WIDTH / 2, self.pad2_Y - PAD_HEIGHT / 2))


        terminal = False
        reward1 = 0
        reward2 = 0
        # score or not
        if self.ball_X < BALL_SIZE/2:
            reward1 = PENALTY
            reward2 = REWARD
            terminal = True
            #self.total_score_1 += reward1
            self.total_score_2 += reward2
            total_score_1 = self.total_score_1
            total_score_2 = self.total_score_2
            two_players = self.two_players
            pad1_Y = self.pad1_Y
            pad2_Y = self.pad2_Y
            self.__init__(two_players)
            self.total_score_1 = total_score_1
            self.total_score_2 = total_score_2
            if max(self.total_score_1, self.total_score_2) >= 210:
                self.total_score_1 = 0
                self.total_score_2 = 0
            else:
                self.pad1_Y = pad1_Y
                self.pad2_Y = pad2_Y

        elif self.ball_X > SCREENWIDTH - BALL_SIZE/2:
            reward1 = REWARD
            reward2 = PENALTY
            terminal = True
            self.total_score_1 += reward1
            #self.total_score_2 += reward2
            total_score_1 = self.total_score_1
            total_score_2 = self.total_score_2
            two_players = self.two_players
            pad1_Y = self.pad1_Y
            pad2_Y = self.pad2_Y
            self.__init__(two_players)
            self.total_score_1 = total_score_1
            self.total_score_2 = total_score_2
            if max(self.total_score_1, self.total_score_2) >= 210:
                self.total_score_1 = 0
                self.total_score_2 = 0
            else:
                self.pad1_Y = pad1_Y
                self.pad2_Y = pad2_Y

        # Update the view
        pygame.display.update()
        # Update time
        FPSCLOCK.tick(FPS)

        return [state1, state2], [reward1, reward2], terminal, [self.total_score_1, self.total_score_2]


def human_play():
    game = Pong(two_players=True)
    t = 0
    while 1:
        action = [0,0]
        if pygame.key.get_pressed()[pygame.K_w]:
            action[0] = 1
        elif pygame.key.get_pressed()[pygame.K_s]:
            action[0] = 2
        else:
            action[0] = 0

        if pygame.key.get_pressed()[pygame.K_UP]:
            action[1] = 1
        elif pygame.key.get_pressed()[pygame.K_DOWN]:
            action[1] = 2
        else:
            action[1] = 0
        pygame.event.poll()
        states, rewards, terminal, scores = game.step(action)
        print ('reward: '+str(rewards))
        print ('terminal: '+str(terminal))
        print ('scores: '+str(scores))
        plt.subplot(1, 2, 1)
        plt.imsave('pong1.jpg', states[0])
        plt.imshow(skimage.color.rgb2grey(skimage.transform.resize(states[0], (80, 80))), cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imsave('pong2.jpg', states[1])
        plt.imshow(skimage.color.rgb2grey(skimage.transform.resize(states[1], (80, 80))), cmap='gray')
        plt.show()
        t+=1

if __name__ == "__main__":
    human_play()




