import pygame
"""
This script loads all the images for every elements in the game: background, hunters, preys.
"""

def load():

    IMAGES = {}

    IMAGES['prey_self'] = pygame.image.load('prey_self.png').convert_alpha()

    IMAGES['prey_other'] = pygame.image.load('prey_other.png').convert_alpha()

    IMAGES['background'] = pygame.image.load('background_pong.png').convert_alpha()

    IMAGES['hunter'] = pygame.image.load('hunter2.png').convert_alpha()

    IMAGES['bonus'] = pygame.image.load('bonus.jpg').convert_alpha()

    return IMAGES