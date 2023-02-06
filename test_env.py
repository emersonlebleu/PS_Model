import os
import pygame as pg

if not pg.font:
    print('Warning, fonts disabled')
if not pg.mixer:
    print('Warning, sound disabled')

main_dir = os.path.split(os.path.abspath(__file__))[0]
stimuli_dir = os.path.join(main_dir, 'stimuli')

def load_image(file_name, colorkey=None, scale=1):
    file_path = os.path.join(stimuli_dir, file_name)

    try:
        image = pg.image.load(file_path)
    except pg.error:
        raise SystemExit('Could not load image "%s" %s'%(file, pg.get_error()))

    size = image.get_size()
    size = (size[0] * scale, size[1] * scale)
    image = pg.transform.scale(image, size)

    image = image.convert()
    if colorkey is not None:
        if colorkey == -1:
            colorkey = image.get_at((0,0))
        image.set_colorkey(colorkey, pg.RLEACCEL)
    
    return image, image.get_rect()

class Box(pg.sprite.Sprite):
    def __init__(self, stimuli):
        pg.sprite.Sprite.__init__(self)
        self.image, self.rect = load_image(stimuli)
        self.size = (300, 300)
        self.rect.topleft = (screen.get_width() / 2 - self.rect.width / 2, screen.get_height() / 2 - self.rect.height / 2)

    def shrink(self):
        if self.size == (0, 0):
            return
        self.size = (self.size[0] - 30, self.size[1] - 30)
        self.image = pg.transform.scale(self.image, self.size)
        self.rect = self.image.get_rect()
        self.rect.topleft = self.get_center_position()
        # print(self.size)

    def grow(self):
        if self.size == (0, 0):
            self.image, self.rect = load_image(stimuli)
            self.size = (30, 30)
            self.image = pg.transform.scale(self.image, self.size)
            self.rect = self.image.get_rect()
            self.rect.topleft = self.get_center_position()
            return
        if self.size == (600, 600):
            return
        self.size = (self.size[0] + 30, self.size[1] + 30)
        self.image = pg.transform.scale(self.image, self.size)
        self.rect = self.image.get_rect()
        self.rect.topleft = self.get_center_position()
        # print(self.size)

    def get_center_position(self):
        self.center_position = (screen.get_width() / 2 - self.rect.width / 2, screen.get_height() / 2 - self.rect.height / 2)
        return self.center_position

pg.init()
screen = pg.display.set_mode((1200, 900), flags=pg.SCALED)
pg.display.set_caption('Boxelerate')
pg.mouse.set_visible(1)

background = pg.Surface(screen.get_size())
background = background.convert()
background.fill((255, 255, 255))

screen.blit(background, (0, 0))
pg.display.flip()

box = Box('green_box.png')
allSprites = pg.sprite.RenderPlain((box))
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
                box.grow()
            elif event.key == pg.K_DOWN:
                box.shrink()
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