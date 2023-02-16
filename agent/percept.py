import os
import pygame as ps
import numpy as np

class Percept:
    def __init__(self, **kwargs):
        # these are the current possible dimensions of a percept this could have been represented as a list 
        # but I wanted to make it more explicit if the dimensionality were greater than 2 we could use a list/tuple or array
        if 'size' in kwargs:
            self.size = kwargs['size']
        
        if 'color' in kwargs:
            self.color = kwargs['color']