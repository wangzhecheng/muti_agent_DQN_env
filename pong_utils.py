import pygame
"""
This script loads all the images for every elements in the Pong
"""

def load():

    IMAGES = {}

    IMAGES['paddle_self'] = pygame.image.load('paddle_self.png').convert_alpha()

    IMAGES['paddle_other'] = pygame.image.load('paddle_other.png').convert_alpha()

    IMAGES['background'] = pygame.image.load('background_pong.png').convert_alpha()

    IMAGES['ball'] = pygame.image.load('ball.png').convert_alpha()

    return IMAGES