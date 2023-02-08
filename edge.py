import os
import pygame as ps
import numpy as np

class Edge:
    def __init__(self, weight=1, emotion=0):
        self.weight = weight
        self.emotion = emotion