from visualizationSprite import VisualizationSprite
from visualization import PLAYER_NUM, ENEMY_NUM, Visualization
from enemy import ENEMY_COLOR, Enemy
from pygame import surface

import pygame
from player import DO_NOTHING_ACTION, PLAYER_COLOR, Player

from environment import Environment

from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    K_m,
    K_n,
    KEYUP,
    QUIT,
)

from pygame import Surface


#####################################################
# Used the following tutorial in building the game: #
# https://realpython.com/pygame-a-primer/           #
#####################################################

class Game():
    screen = None
    running = True

    def initialize_game(self):

        # Keep track of if the game is over
        self.lost = False

        # Keep track of dead enemies
        self.passed_enemies=False


        # Setup resetable part of fps clock
        self.clock = pygame.time.Clock()
        self.delta_time = 1

        # Create player
        self.player = Player(75, 75, (self.width-75)/2, (self.height-75)/2, 10*self.speed)

        # Setup Score
        self.score = 0
        self.SCORE_INCREMENT = 1

        # Adding enemy event
        self.SCORE_BETWEEN_ENEMIES = 15
        self.ADD_ENEMY_EVENT_NUM = pygame.USEREVENT + 1

        # Adding all_sprites group for rendering
        self.all_sprites = pygame.sprite.Group()
        self.all_sprites.add(self.player)

        # Adding enemies group for collision checking
        self.enemies = pygame.sprite.Group()

    def reset(self):
        # Re-initializes the game
        self.initialize_game()

    def __init__(self, width,  height, fps, title, vis_x_points, vis_y_points, vis_x_cor, vis_y_cor, vis_pixel_size, display_visualization=True, slow_down_game=True, speed=1):

        # Initialize pygame
        pygame.init()

        self.speed = speed

        # Keep track of game number
        self.game_num = 0

        # Slow down the game
        self.slow_down_game = slow_down_game

        # Set window title
        pygame.display.set_caption(title)

        # Save width and height
        self.width = width
        self.height = height

        # Create the screen
        self.screen = pygame.display.set_mode([width, height])
        
        # Set font that we will use
        self.font = pygame.font.SysFont("Arial", 18)

        # FPS setting
        self.fps = (-1, fps)[self.slow_down_game]
        
        # Adding visualization

        self.display_visualization = display_visualization

        self.vis_x_points = vis_x_points
        self.vis_y_points = vis_y_points

        self.vis_x_cor = vis_x_cor
        self.vis_y_cor = vis_y_cor
        self.vis_pixel_size = vis_pixel_size

        self.visualization = Visualization(self)
        self.visualization_data = self.visualization.get_data(self)

        # # Initialize AI Agent
        # self.environment = Environment(self)

        # Initializes the game
        self.initialize_game()

        # Run the game (do nothing)
        self.run(DO_NOTHING_ACTION) 

    def run(self, action):

        if (not self.lost):

            # Updates the events
            self.update(action)

            # Updates the screen
            self.paint()

            # # Wait for agent to make its move
            # self.environment.step(self)

        if(not self.running):
            # game finished!
            pygame.quit()

    def update(self, action):

        # check if enemy event should be triggered
        if (self.score % self.SCORE_BETWEEN_ENEMIES == 0):
            pygame.event.post(pygame.event.Event(self.ADD_ENEMY_EVENT_NUM, message="Enemy Added!"))

        for event in pygame.event.get():

                # Check if the window wants to quit
                if event.type == QUIT:
                    self.running = False

                elif event.type == KEYUP:
                    if event.key == K_m:
                        self.fps += 10
                    if event.key == K_n:
                        self.fps -= 10

                # check for adding enemy event
                elif event.type == self.ADD_ENEMY_EVENT_NUM:
                    # Create enemy
                    new_enemy = Enemy(40, 40, 5*self.speed, 10*self.speed, 20, 100, self)
                    self.enemies.add(new_enemy)
                    self.all_sprites.add(new_enemy)

        # # Get all keys that are currently pressed
        # pressed_keys = pygame.key.get_pressed()


        # # move player first
        # self.player.update(pressed_keys, self)

        self.player.update(action, self)

        # Update enemies
        for enemy in self.enemies:
            enemy.update(self)

        # Check if there was any entity collisions
        if pygame.sprite.spritecollideany(self.player, self.enemies):
            # If yes, game over!
            self.lost = True
        else:
            # Add to score!
            self.score+=self.SCORE_INCREMENT

    def paint(self):
        # black background
        self.screen.fill((0,0,0))

        # Display all entities
        for entity in self.all_sprites:
            self.screen.blit(entity.surf, entity.rect)
    
        # Display fps
        self.screen.blit(self.get_fps(), (5, 0))

        # Display score
        self.screen.blit(self.get_score(), (5, 20))


        if(self.display_visualization):
            # Display visualization
            for i in range(len(self.visualization_data)):
                for j in range(len(self.visualization_data[i])):

                    

                    tmpObj = None

                    if(self.visualization_data[i][j] == PLAYER_NUM):
                        tmpObj = VisualizationSprite(self.vis_pixel_size, self.vis_pixel_size, (self.vis_x_cor + i*self.vis_pixel_size), (self.vis_y_cor + j*self.vis_pixel_size), PLAYER_COLOR)

                    elif (self.visualization_data[i][j] == ENEMY_NUM):
                        tmpObj = VisualizationSprite(self.vis_pixel_size, self.vis_pixel_size, (self.vis_x_cor + i*self.vis_pixel_size), (self.vis_y_cor + j*self.vis_pixel_size), ENEMY_COLOR)
                        pass
                    if (tmpObj != None):
                        self.screen.blit(tmpObj.surf, tmpObj.rect)

        # Flip buffer
        pygame.display.flip()

        # Update visualization
        self.visualization_data = self.visualization.get_data(self)

    def get_fps(self):

        self.measured_fps = int(self.clock.get_fps())
        fps = str(self.measured_fps)
        fps_text = self.font.render("FPS: " + fps + "/" + str(self.fps), True, pygame.Color("coral"))
        return fps_text

    def get_score(self):
        score_text = self.font.render("Score: " + str(self.score), True, pygame.Color("white"))
        return score_text
        


# if __name__ == "__main__":
#     global gameInstance
#     gameInstance = Game(1280, 720, 60, "Simple Ai Game", int(1280/10), int(720/10), 100, 10, 1, display_visualization=True, slow_down_game=True)
