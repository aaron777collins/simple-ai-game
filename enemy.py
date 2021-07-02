import pygame

from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    KEYDOWN,
    QUIT,
)

from pygame import Surface

import random

ENEMY_COLOR = pygame.Color("red")

class Enemy(pygame.sprite.Sprite):
    def __init__(self, width, height, min_speed, max_speed, min_dist, max_dist, game):
        pygame.sprite.Sprite.__init__(Enemy, self)
        self.surf = Surface((width, height))
        self.surf.fill(ENEMY_COLOR)
        self.rect = self.surf.get_rect(
            center=(
                random.randint(game.width + min_dist, game.width + max_dist),
                random.randint(0, game.height),
            )
        )
        self.speed = speed = random.uniform(min_speed, max_speed)

    def update(self, game):
        self.rect.move_ip(-self.speed, 0)
    
        self.check_bounds(game)

    def check_bounds(self, game):
        if self.rect.right < 0:
            try:
                self.kill()
            except:
                print("error killing rect")
