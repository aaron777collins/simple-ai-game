from enemy import ENEMY_COLOR
from player import PLAYER_COLOR

PLAYER_NUM = -1
ENEMY_NUM = 1

class Visualization():
    def __init__(self, game):
        self.data = [[0 for y in range(game.vis_y_points)] for x in range(game.vis_x_points)]

    def get_data(self, game):
        self.data = [[0 for y in range(game.vis_y_points)] for x in range(game.vis_x_points)]
        for i in range(game.vis_x_points):
            for j in range(game.vis_y_points):

                color = game.screen.get_at((i * int(game.width/game.vis_x_points),(j * int(game.height/game.vis_y_points))))                

                if color == ENEMY_COLOR:
                    self.data[i][j] = 1

                if color == PLAYER_COLOR:
                    self.data[i][j] = -1                    

        return self.data

    
    