import pygame

from pygame import Surface
class VisualizationSprite(pygame.sprite.Sprite):
    def __init__(self, width, height, x, y, color):
        pygame.sprite.Sprite.__init__(VisualizationSprite, self)
        self.surf = Surface((width, height))
        self.surf.fill(color)
        self.rect = self.surf.get_rect()
        self.rect.x = x
        self.rect.y = y

