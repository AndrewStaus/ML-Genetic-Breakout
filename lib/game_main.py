import os
from numpy.core.numeric import Infinity
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
import pygame
import numpy as np
from lib.game_objects import *
from lib.game_handlers import *


INITIAL_LIVES = 3
INITIAL_LAYERS = 3
MAX_LEVELS = 15
FPS = 120

SCREEN_WIDTH, SCREEN_HEIGHT = 600, 800
BLOCK_WIDTH, BLOCK_HEIGHT = 50, 20
PROJECTILE_SIZE = 10
BLACK = (0, 0, 0)

game = Game(FPS, INITIAL_LIVES, INITIAL_LAYERS, MAX_LEVELS, SCREEN_WIDTH, SCREEN_HEIGHT)
paddle = Paddle(SCREEN_WIDTH // 2 - game.paddle_width // 2,
                SCREEN_HEIGHT - 10 - 30,
                game.paddle_width,
                10)
projectile = Projectile(paddle, game, PROJECTILE_SIZE)


def setup(newgame:bool = False):

    if newgame:
        game.reset()
        paddle.reset()

    else:
        game.changeBackground()
        game.blocks = []

    projectile.reset()

    for j in range(game.MAX_LEVELS):
        layer_color = random_color(0, 255)
        for i in range(SCREEN_WIDTH//BLOCK_WIDTH):
            if j+1 < game.level + INITIAL_LAYERS:
                block = Block(True, layer_color, i*BLOCK_WIDTH, j*BLOCK_HEIGHT + 100 + j*5, BLOCK_WIDTH, BLOCK_HEIGHT)
                game.blocks.append(block)
            else:
                block = Block(False, layer_color, i*BLOCK_WIDTH, j*BLOCK_HEIGHT + 100 + j*5, BLOCK_WIDTH, BLOCK_HEIGHT)
                game.blocks.append(block)

def draw_window(screen, myfont):

    screen.fill(game.background)
    scoreboard = myfont.render(str(game.score), False, (0, 0, 0))
    livesboard = myfont.render(f'LIVES: {game.lives}', False, (0, 0, 0))
    levelboard = myfont.render(f'LEVEL: {game.level}', False, (0, 0, 0))
 

    screen.blit(scoreboard,(SCREEN_WIDTH // 2 - scoreboard.get_width() // 2 ,5))
    screen.blit(livesboard,(0,5))
    screen.blit(levelboard,(SCREEN_WIDTH - levelboard.get_width(),5))


    pygame.draw.rect(screen, BLACK, paddle)
    pygame.draw.rect(screen, BLACK, projectile)

    for block in game.blocks:
        if block.active:
            pygame.draw.rect(screen, block.COLOR, block, border_radius=10)

    pygame.display.update()

def output():

    out_1= paddle.x / SCREEN_WIDTH
    out_2 = projectile.x / SCREEN_WIDTH
    out_3 = projectile.y / SCREEN_HEIGHT
    out_4 = (projectile.x_direction + 1) / 2
    out_5 = (projectile.y_direction + 1) / 2

    output = [out_1, out_2, out_3, out_4, out_5] + game.getBlockStatus() 
    output = output 
    output = np.array(output)
    return output



def main(graphics = True, agent= None):

    
    pygame.init()


    if graphics:
        pygame.display.set_caption("Breakout")
        pygame.font.init()
        myfont = pygame.font.Font(r'.\lib\font\font.ttf', 20)
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        game.FPS = FPS
        

    setup(newgame=True)

    if agent:
        game.lives = 1
        game.FPS = Infinity

    clock = pygame.time.Clock()
    run = True
    frames_since_last_point = 0
    while run:

        
        clock.tick(game.FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if event.type == SCORE:
                game.score += 1
                frames_since_last_point = 0

            if event.type == DEATH:
                game.lives -= 1
                frames_since_last_point = 0



        keys_pressed = pygame.key.get_pressed()
        handle_input(keys_pressed, paddle, projectile, game)

        if agent:
            softmax = agent(output())
            network_inputs = [softmax == np.max(softmax)][0]
            handle_input(keys_pressed, paddle, projectile, game, network_inputs=network_inputs)
            if frames_since_last_point >= 5000:
                run = False
                return(game.score)

            frames_since_last_point += 1

        if projectile.fired:
            handle_colission(projectile, paddle, game)
            projectile.move()

        if game.lives <= 0:
            if agent:
                run=False
                return(game.score)
            else:
                setup(newgame = True)
                

        if game.level >= MAX_LEVELS +1:
            if agent:
                run=False
                return(game.score*4)

            print(f'Game Over!  Score: {game.score}')
            game.score = 0
            setup(newgame=True)


        if game.activeBlocks() <= 0:
            game.level += 1
            setup()


        if graphics:
            draw_window(screen, myfont)


    pygame.quit()
