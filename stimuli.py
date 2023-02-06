import os
import pygame as pg

main_dir = os.path.split(os.path.abspath(__file__))[0]
stimuli_dir = os.path.join(main_dir, 'stimuli')

# A function to load images
def load_image(file_name: str, colorkey=None, scale=1):

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

# A class for the stimuli
class Stimuli(pg.sprite.Sprite):
    def __init__(self, stimuli_name: str):
        pg.sprite.Sprite.__init__(self)
        self.stimuli_name = stimuli_name

        self.image, self.rect = load_image(self.stimuli_name)
        self.size = (300, 300)
        self.rect.topleft = self.get_center_position()

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
        self.center_position = (pg.display.get_window_size()[0] / 2 - self.rect.width / 2, pg.display.get_window_size()[1] / 2 - self.rect.height / 2)
        return self.center_position

