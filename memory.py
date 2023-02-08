import os
import pygame as ps
import numpy as np

class Memory:
    def __init__(self):
        self.clips = {}

    def add_clip(self, clip):
        self.clips[clip.percept] = clip.connections