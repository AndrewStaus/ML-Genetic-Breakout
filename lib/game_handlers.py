"""
Event handlers for the game
Handles user and agent inputs, as well as game events.
"""
import pygame

SCORE = pygame.USEREVENT + 1
DEATH = pygame.USEREVENT + 2

def handle_input(keys_pressed, paddle, projectile, game, network_inputs = []):
        if len(network_inputs):
            network_space = 1
            network_left = network_inputs[0]
            network_right = network_inputs[1]
        else:
            network_space = 0
            network_left = 0
            network_right = 0

        if (keys_pressed[pygame.K_SPACE] or network_space) and projectile.fired == False:
            projectile.fired = True

        if (keys_pressed[pygame.K_a] or keys_pressed[pygame.K_LEFT] or network_left) and paddle.x - game.paddle_speed > -10: #LEFT
            paddle.x -= game.paddle_speed
            if projectile.fired == False:
                projectile.x -= game.paddle_speed

        if (keys_pressed[pygame.K_d] or keys_pressed[pygame.K_RIGHT] or network_right) and paddle.x + paddle.width + game.paddle_speed < game.SCREEN_WIDTH + 10: #RIGHT
            paddle.x += game.paddle_speed
            if projectile.fired == False:
                projectile.x += game.paddle_speed


def handle_colission(projectile, paddle, game):
    for block in game.blocks:
        if projectile.colliderect(block) and block.active:
            block.active = False
            projectile.y_direction = projectile.y_direction * -1
            pygame.event.post(pygame.event.Event(SCORE))

    if projectile.colliderect(paddle):

        paddle_center = paddle.x + game.paddle_width//2
        projectile_center = projectile.x + projectile.projectile_size//2
        vector = projectile_center-paddle_center
        if abs(vector) < 50:
            vector = 60

        projectile.x_direction = vector / 50
        projectile.y_direction = projectile.y_direction*-1

    #projectile colides with wall
    if projectile.x > game.SCREEN_WIDTH or projectile.x < 0:
        projectile.x_direction = projectile.x_direction * -1

    #projectile colides with roof
    if projectile.y < 0:
        projectile.y_direction = projectile.y_direction * -1

    #projectile colides with floor (death)
    if projectile.y > game.SCREEN_HEIGHT:
        projectile.reset()
        pygame.event.post(pygame.event.Event(DEATH))