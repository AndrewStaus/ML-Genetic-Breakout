"""# Breakout Game Classes"""

import pygame
import random
import numpy as np

def random_color(min, max):
    return (random.randint(min,max),random.randint(min,max),random.randint(min,max))

class Game:
    def __init__(self, FPS, INITIAL_LIVES, INITIAL_LAYERS, MAX_LEVELS, SCREEN_WIDTH, SCREEN_HEIGHT):
        self.FPS = FPS
        self.INITIAL_LIVES = INITIAL_LIVES
        self.INITIAL_LAYERS = INITIAL_LAYERS
        self.MAX_LEVELS = MAX_LEVELS
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.background = random_color(175,200)
        self.score = 0
        self.lives = INITIAL_LIVES
        self.level = 1
        self.paddle_width = 120
        self.paddle_speed = 10
        self.projectile_speed = 7
        self.block_layers = INITIAL_LAYERS
        self.blocks = []

    def reset(self):
        self = self.__init__(self.FPS, self.INITIAL_LIVES, self.INITIAL_LAYERS, self.MAX_LEVELS, self.SCREEN_WIDTH, self.SCREEN_HEIGHT)


    def changeBackground(self):
        self.BACKGROUND = random_color(175,200)


    def activeBlocks(self):
        active_blocks = 0
        for block in self.blocks:
            if block.active:
                active_blocks += 1
        return active_blocks


    def getBlockStatus(self):
        status = [int(block.active) for block in self.blocks]
        block_values = []
        for i in range(12):
            block_row=[]
            for j in range(20):
                block_row.append(status[i+j])
            block_values.append(block_row)

        block_values = np.array(block_values)

        return list(block_values.sum(axis=0) / 20)


class Block(pygame.Rect):

    def __init__(self, active, color:tuple, *args, **kwargs):
        self.active = active
        self.COLOR = color
        super().__init__(*args, **kwargs)


class Paddle(pygame.Rect):
    def __init__(self, left:float, top: float, width: float, height: float):
        self.initial_left = left
 
        super().__init__(left, top, width, height)


    def reset(self):
        self.x = self.initial_left


class Projectile(pygame.Rect):
    def __init__(self, paddle, game, projectile_size):
        self.game = game
        self.paddle = paddle
        self.projectile_size = projectile_size
        self.reset()
        super().__init__(paddle.x + paddle.width//2 - self.projectile_size//2, paddle.y-paddle.height-10, self.projectile_size, self.projectile_size)


    def move(self):
        self.y -= self.game.projectile_speed * self.y_direction
        self.x -= self.game.projectile_speed * -self.x_direction


    def reset(self):
            self.y_direction = 1
            #self.x_direction = 0.5 * 60 /self.game.FPS
            self.x_direction = 0.25
            self.x = self.paddle.x + self.game.paddle_width//2 - self.projectile_size//2
            self.y = self.paddle.y-self.paddle.height-10
            self.fired = False