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

PLAYER_COLOR = pygame.Color("white")
DO_NOTHING_ACTION = 8

from pygame import Surface
class Player(pygame.sprite.Sprite):
    def __init__(self, width, height, x, y, speed):
        pygame.sprite.Sprite.__init__(Player, self)
        self.surf = Surface((width, height))
        self.surf.fill(PLAYER_COLOR)
        self.rect = self.surf.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.speed = speed

    # def update(self, pressed_keys, game):
    #     if pressed_keys[K_UP]:
    #         self.rect.move_ip(0, -self.speed)
    #     if pressed_keys[K_DOWN]:
    #         self.rect.move_ip(0, self.speed)
    #     if pressed_keys[K_LEFT]:
    #         self.rect.move_ip(-self.speed, 0)
    #     if pressed_keys[K_RIGHT]:
    #         self.rect.move_ip(self.speed, 0)

    #     self.check_bounds(game)


    def update(self, action, game):

        if action == 0:
            self.rect.move_ip(-self.speed, -self.speed)
        elif action == 1:
            self.rect.move_ip(0, -self.speed),
        elif action == 2:
            self.rect.move_ip(self.speed, -self.speed),
        elif action == 3:
            self.rect.move_ip(self.speed, 0),
        elif action == 4:
            self.rect.move_ip(self.speed, self.speed),
        elif action == 5:
            self.rect.move_ip(0, self.speed),
        elif action == 6:
            self.rect.move_ip(-self.speed, self.speed),
        elif action == 7:
            self.rect.move_ip(-self.speed, 0),
        elif action == DO_NOTHING_ACTION:
            pass

        self.check_bounds(game)

    def check_bounds(self, game):
        if (self.rect.left < 0):
            self.rect.left = 0
        if (self.rect.right > game.width):
            self.rect.right = game.width
        if (self.rect.top < 0):
            self.rect.top = 0
        if (self.rect.bottom > game.height):
            self.rect.bottom = game.height

