import os
import pygame as ps
import numpy as np

class Percept:
    def __init__(self, **kwargs):
        if 'size' in kwargs:
            self.size = kwargs['size']
        
        if 'color' in kwargs:
            self.color = kwargs['color']