import os
import pygame as pg
from stimuli import Stimuli

if not pg.font:
    print('Warning, fonts disabled')
if not pg.mixer:
    print('Warning, sound disabled')

# Start the game
pg.init()
screen = pg.display.set_mode((1200, 900), flags=pg.SCALED)
pg.display.set_caption('Boxelerate')
pg.mouse.set_visible(1)

background = pg.Surface(screen.get_size())
background = background.convert()
background.fill((255, 255, 255))

screen.blit(background, (0, 0))
pg.display.flip()

# Create the stimuli
stimuli = Stimuli('green_box.png')

allSprites = pg.sprite.RenderPlain((stimuli))
clock = pg.time.Clock()

quit = False

while not quit:
    clock.tick(60)

    for event in pg.event.get():
        if event.type == pg.QUIT:
            quit = True
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_ESCAPE:
                quit = True
            elif event.key == pg.K_UP:
                stimuli.grow()
            elif event.key == pg.K_DOWN:
                stimuli.shrink()
        elif event.type == pg.VIDEORESIZE:
            screen = pg.display.set_mode(event.size, pg.RESIZABLE)
            background = pg.Surface(screen.get_size())
            background = background.convert()
            background.fill((255, 255, 255))
            screen.blit(background, (0, 0))
            pg.display.flip()

    allSprites.update()
    screen.blit(background, (0,0))
    allSprites.draw(screen)
    pg.display.flip()

pg.quit()